# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Dict, List, Optional, Tuple, Union
import hydra
from pytorch_lightning import LightningModule
import torch
from omegaconf import DictConfig
from torch import Tensor, nn
import wandb
from pathlib import Path
from collections import OrderedDict

from models.metrics.training import TrainingMetrics
from models.metrics.logging import ErrorMetrics, TrafficRuleMetrics
from models.modules.distributions import MyDist
from models.modules.action_head import ActionHead
from utils.traffic_rule_checker import TrafficRuleChecker
from utils.teacher_forcing import TeacherForcing
from utils.buffer import RolloutBuffer
from utils.rewards import DifferentiableReward
from utils.dynamics import Dynamics
from utils.vis_waymo import VisWaymo
from data_modules.waymo_post_processing import WaymoPostProcessing
from models.metrics.womd import WOMDMetrics
from utils.submission import SubWOMD


class WaymoMotion(LightningModule):
    def __init__(
        self,
        time_step_current: int,
        time_step_gt: int,
        time_step_end: int,
        time_step_sim_start: int,
        hidden_dim: int,
        data_size: DictConfig,
        pre_processing: DictConfig,
        step_detach_hidden: int,
        model: DictConfig,
        p_training_rollout_prior: float,
        detach_state_policy: bool,
        training_deterministic_action: bool,
        differentiable_reward: DictConfig,
        p_drop_hidden: float,
        n_video_batch: int,
        n_joint_future: int,
        waymo_post_processing: DictConfig,
        dynamics: DictConfig,
        action_head: DictConfig,
        teacher_forcing_training: DictConfig,
        teacher_forcing_reactive_replay: DictConfig,
        teacher_forcing_joint_future_pred: DictConfig,
        training_metrics: DictConfig,
        traffic_rule_checker: DictConfig,
        optimizer: DictConfig,
        lr_scheduler: Optional[DictConfig],
        lr_goal: float,
        sub_womd_reactive_replay: DictConfig,
        sub_womd_joint_future_pred: DictConfig,
        interactive_challenge: bool = False,
        wb_artifact: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # pre_processing
        self.pre_processing = []
        pre_proc_kwargs = {}
        for k, v in pre_processing.items():
            _pre_proc = hydra.utils.instantiate(v, time_step_current=time_step_current, data_size=data_size)
            self.pre_processing.append((k, _pre_proc))
            pre_proc_kwargs |= _pre_proc.model_kwargs
        self.pre_processing = nn.Sequential(OrderedDict(self.pre_processing))

        # model
        self.model = hydra.utils.instantiate(model, **pre_proc_kwargs, _recursive_=False)
        self.dynamics = Dynamics(**dynamics)
        self.action_head = ActionHead(hidden_dim=hidden_dim, action_dim=self.dynamics.action_dim, **action_head)

        # diffbar rewards
        self.diffbar_reward = DifferentiableReward(**differentiable_reward)

        # training
        self.teacher_forcing_training = TeacherForcing(**teacher_forcing_training)
        self.train_metrics_train = TrainingMetrics("training", **training_metrics)

        # reactive_replay: scene reconstruct given a complete episode, same setup as training
        self.teacher_forcing_reactive_replay = TeacherForcing(**teacher_forcing_reactive_replay)
        self.train_metrics_reactive_replay = TrainingMetrics("reactive_replay", **training_metrics)
        self.err_metrics_reactive_replay = ErrorMetrics("reactive_replay")
        self.rule_metrics_reactive_replay = TrafficRuleMetrics("reactive_replay")
        self.waymo_post_processing = WaymoPostProcessing(**waymo_post_processing)
        self.womd_metrics_reactive_replay = WOMDMetrics(
            "reactive_replay", time_step_end, time_step_current, interactive_challenge
        )

        # joint_future_pred: no spawn, tl_state from current_step, prior latent, goal/dest
        self.teacher_forcing_joint_future_pred = TeacherForcing(**teacher_forcing_joint_future_pred)
        self.err_metrics_joint_future_pred = ErrorMetrics("joint_future_pred")
        self.rule_metrics_joint_future_pred = TrafficRuleMetrics("joint_future_pred")
        self.womd_metrics_joint_future_pred = WOMDMetrics(
            "joint_future_pred", time_step_end, time_step_current, interactive_challenge
        )

        # save submission files
        self.sub_womd_reactive_replay = SubWOMD(wb_artifact=wb_artifact, **sub_womd_reactive_replay)
        self.sub_womd_joint_future_pred = SubWOMD(wb_artifact=wb_artifact, **sub_womd_joint_future_pred)

    def forward(
        self,
        map_feature: Tensor,
        map_valid: Tensor,
        tl_feature: Tensor,
        tl_valid: Tensor,
        goal_feature: Tensor,
        goal_valid: Tensor,
        action_override: Optional[Tensor] = None,
        mask_action_override: Optional[Tensor] = None,
        state_override: Optional[Dict[str, Tensor]] = None,
        mask_state_override: Optional[Tensor] = None,
        deterministic_action: bool = True,
        require_train_dict: bool = True,
        require_vis_dict: bool = False,
    ) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
        """
        Args:
            map_feature, map_valid: [n_batch, n_pl, (hidden_dim)]
            tl_feature, tl_valid: [n_batch, n_tl, (hidden_dim)]
            action_override: [n_batch, n_agent, k_pred, 2]
            mask_action_override: [n_batch, n_agent] bool
            state_override["as_pred"]: [n_batch, n_agent, k_pred, 4]
            state_override["acc"], state_override["yaw_rate]: [n_batch, n_agent, 1]
            mask_state_override: [n_batch, n_agent] bool, can be used to spawn new agents

        Return:
            state: [n_batch, n_agent, 4], x,y,theta,v
            valid: [n_batch, n_agent], bool
        """
        agent_valid = self.dynamics.agent_valid
        agent_state = self.dynamics.agent_state
        agent_attr, agent_pe = self.pre_processing.input.get_agent_attr_and_pe(
            agent_pos=agent_state[:, :, :2],
            agent_yaw_bbox=agent_state[:, :, 2:3],
            agent_spd=agent_state[:, :, 3:4],
            agent_vel=self.dynamics.vel,
            agent_yaw_rate=self.dynamics.yaw_rate,
            agent_acc=self.dynamics.acc,
            agent_size=self.dynamics.agent_size,
            agent_type=self.dynamics.agent_type,
        )
        if self.hparams.detach_state_policy:
            agent_state = agent_state.detach()
            agent_attr = agent_attr.detach()
            agent_pe = agent_pe.detach()

        agent_feature = self.model.agent_encoder(agent_valid, agent_attr, agent_pe)

        # some policies (e.g. TrafficSim) will infer a new pair of latent_post and latent_prior at each step.
        policy_feature, latent_logp, attn_pl, attn_tl, attn_agent = self.model(
            agent_valid=agent_valid,
            agent_feature=agent_feature,
            map_valid=map_valid,
            map_feature=map_feature,
            tl_valid=tl_valid,
            tl_feature=tl_feature,
            goal_valid=goal_valid,
            goal_feature=goal_feature,
            need_weights=require_vis_dict,
        )

        # action_dist: Gaussian or DiagGaussian
        action_dist = self.action_head(policy_feature, agent_valid, self.dynamics.agent_type)
        pred_state, pred_valid, action, action_log_prob = self.dynamics.update(
            action_dist=action_dist,
            action_override=action_override,
            mask_action_override=mask_action_override,
            deterministic=deterministic_action,
        )
        self.dynamics.override_states(state_override=state_override, mask_state_override=mask_state_override)

        if require_train_dict:
            train_dict = {
                # for logging
                "latent_log_prob": latent_logp,  # [n_batch, n_agent]
                "action_log_prob": action_log_prob,  # [n_batch, n_agent]
                # for train_metrics
                "pred_valid": pred_valid,  # old agent_valid, not updated with mask_override
                "pred_state": pred_state,
            }
        else:
            train_dict = {}
        if require_vis_dict:
            if goal_valid is None:
                goal_valid = torch.zeros_like(pred_valid)
            vis_dict = {
                "action": action.cpu(),
                "goal_valid": goal_valid.cpu(),
                "attn_weights_to_pl": attn_pl.cpu(),
                "attn_weights_to_tl": attn_tl.cpu(),
                "attn_weights_to_agent": attn_agent.cpu(),
            }
        else:
            vis_dict = {}
        return self.dynamics.agent_state, self.dynamics.agent_valid, train_dict, vis_dict

    def rollout(
        self,
        features: Dict[str, Tensor],
        latent: MyDist,
        goal: Tensor,
        goal_valid: Tensor,
        mask_teacher_forcing: Tensor,
        rule_checker: TrafficRuleChecker,
        deterministic_latent: Union[bool, Tensor],
        deterministic_action: bool,
        step_end: int,
        step_start: int,
        require_vis_dict: bool = False,
        gt_sdc: Optional[Dict[str, Tensor]] = None,
    ):
        """
        Args:
            features: step_current as step_gt during validation
                "map_valid": [n_batch, n_pl] bool
                "map_feature": [n_batch, n_pl, hidden_dim]
                "tl_valid": [n_batch, step_tl_gt+1, n_tl] bool
                "tl_feature": [n_batch, step_tl_gt+1, n_tl, hidden_dim]
                "agent_valid": [n_batch, step_gt+1, n_agent] bool
                "agent_state": [n_batch, step_gt+1, n_agent, 4], x,y,theta,spd
                "agent_type": [n_batch, n_agent, 3] one_hot bool [veh=0, ped=1, cyc=2]
                "agent_size": [n_batch, n_agent, 3] float32 [length, width, height]
                "vel": [n_batch, step_gt+1, n_agent, 2]
                "acc": [n_batch, step_gt+1, n_agent, 1]
                "yaw_rate": [n_batch, step_gt+1, n_agent, 1]
            goal: [n_batch, n_agent, 4] float32[x, y, theta, v] or [n_batch, n_agent] int64: index to map n_pl
            goal_valid: [n_batch, n_agent] bool
            latent: [n_batch, n_agent, hidden_dim]
            mask_teacher_forcing: [n_batch, step_gt+1, n_agent] bool
            deterministic_latent: bool or [n_batch, n_agent] for sampling relevant agents and determistic other agents.
            gt_sdc: None or {"agent_state", "vel", "acc", "yaw_rate"} [n_batch, n_step, 4]

        Returns:
            rollout_buffer [1, ..., step_end]
        """
        assert mask_teacher_forcing.shape[1] == features["agent_state"].shape[1]
        # init buffer and policy
        rollout_buffer = RolloutBuffer(
            step_start=step_start, step_end=step_end, step_current=self.hparams.time_step_current
        )
        self.model.init(latent, deterministic_latent)
        # init with first frame
        self.dynamics.init(
            agent_valid=features["agent_valid"][:, 0],
            agent_state=features["agent_state"][:, 0],
            agent_type=features["agent_type"],
            agent_size=features["agent_size"],
            vel=features["vel"][:, 0],
            acc=features["acc"][:, 0],
            yaw_rate=features["yaw_rate"][:, 0],
        )
        # encode goal
        if self.model.goal_manager.dummy:
            goal_feature = None
        else:
            goal_feature = self.model.goal_manager.get_goal_feature(
                goal=goal, as_state=self.dynamics.agent_state, map_feature=features["map_feature"]
            )

        # rollout
        for _step in range(step_start, step_end + 1):
            # prepare state_override: [n_batch, n_agent]
            if _step >= features["agent_valid"].shape[1]:
                mask_state_override = torch.zeros_like(mask_teacher_forcing[:, 0])
            else:
                mask_state_override = mask_teacher_forcing[:, _step]
            state_override = None
            if mask_state_override.any():
                state_override = {k: features[k][:, _step] for k in self.dynamics.state_keys}
            # what-if motion prediction
            if gt_sdc is not None:
                mask_state_override[:, 0] = True
                if state_override is None:
                    state_override = {k: features[k][:, 0] for k in self.dynamics.state_keys}
                for k in gt_sdc.keys():
                    state_override[k][:, 0] = gt_sdc[k][:, _step]

            # predict _step, given _step-1, using last observered tl_state
            step_tl = min(_step - 1, features["tl_valid"].shape[1] - 1)  # todo(maybe): could also use tl_state=UNKNOWN

            if self.model.goal_manager.update_goal:
                goal_feature = self.model.goal_manager.get_goal_feature(
                    goal=goal, as_state=self.dynamics.agent_state, map_feature=features["map_feature"]
                )

            state_new, valid_new, train_dict, vis_dict = self.forward(
                map_feature=features["map_feature"],
                map_valid=features["map_valid"],
                tl_feature=features["tl_feature"][:, step_tl],
                tl_valid=features["tl_valid"][:, step_tl],
                goal_feature=goal_feature,
                goal_valid=goal_valid,
                state_override=state_override,
                mask_state_override=mask_state_override,
                deterministic_action=deterministic_action,
                require_train_dict=True,
                require_vis_dict=require_vis_dict,
            )

            # check violations, kill agents if outside map, update self.dynamics.agent_valid
            _gt_valid = None if _step >= features["agent_valid"].shape[1] else features["agent_valid"][:, _step]
            _gt_state = None if _step >= features["agent_valid"].shape[1] else features["agent_state"][:, _step]
            violations = rule_checker.check(_step, valid_new, state_new)
            self.dynamics.kill(violations, _gt_valid)
            goal_valid = self.model.goal_manager.disable_goal_reached(
                goal_valid=goal_valid,
                agent_valid=self.dynamics.agent_valid,
                dest_reached=violations["dest_reached"],
                goal_reached=violations["goal_reached"],
            )

            # diffbar_reward
            if self.train_metrics_train.use_diffbar_reward:
                diffbar_reward, diffbar_reward_valid = self.diffbar_reward.get(
                    agent_valid=train_dict["pred_valid"],
                    agent_state=train_dict["pred_state"],
                    gt_valid=_gt_valid,
                    gt_state=_gt_state,
                    agent_size=features["agent_size"],
                )
            else:
                diffbar_reward, diffbar_reward_valid = None, None

            # finish iteration by adding to buffer
            rollout_buffer.add(
                valid=train_dict["pred_valid"],
                pred=train_dict["pred_state"],
                override_mask=mask_state_override,
                violation=violations,
                diffbar_reward=diffbar_reward,
                diffbar_reward_valid=diffbar_reward_valid,
                latent_log_prob=train_dict["latent_log_prob"],
                action_log_prob=train_dict["action_log_prob"],
                vis_dict=vis_dict,
            )

            # stop gradient or dropout during training for hidden
            if self.training and (self.model.hidden is not None):
                if _step <= self.hparams.step_detach_hidden:
                    self.model.hidden = self.model.hidden.detach()
                if self.hparams.p_drop_hidden > 0:
                    if torch.rand(1) < self.hparams.p_drop_hidden:
                        self.model.hidden = torch.zeros_like(self.model.hidden)

        rollout_buffer.finish()
        return rollout_buffer

    def training_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        with torch.no_grad():
            batch = self.pre_processing(batch)
            # for k, v in batch.items():
            #     if not v.is_contiguous():
            #         print(k)
            input_dict = {k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k}
            latent_post_dict = {k.split("latent_post/")[-1]: v for k, v in batch.items() if "latent_post/" in k}
            latent_prior_dict = {k.split("latent_prior/")[-1]: v for k, v in batch.items() if "latent_prior/" in k}

        input_feature_dict = self.model.encode_input_features(**input_dict)
        latent_post_feature_dict = self.model.encode_input_features(**latent_post_dict)
        latent_prior_feature_dict = self.model.encode_input_features(**latent_prior_dict)

        # ! goal and destination
        goal_gt, goal_valid = self.model.goal_manager.get_gt_goal(
            agent_valid=input_dict["agent_valid"], gt_dest=batch["gt/dest"], gt_goal=batch["gt/goal"]
        )
        goal_pred = self.model.goal_manager.pred_goal(
            agent_type=batch["ref/agent_type"],
            map_type=batch["ref/map_type"],
            agent_state=batch["ref/agent_state"],
            **input_feature_dict,
        )

        # ! latents
        latent_post = self.model.latent_encoder(posterior=True, **latent_post_feature_dict)
        latent_prior = self.model.latent_encoder(**latent_prior_feature_dict)
        if torch.rand(1) < self.hparams.p_training_rollout_prior:
            latent = latent_prior
        else:
            latent = latent_post

        # ! rollout
        rollout_buffer = self.reactive_replay(
            batch=batch,
            input_feature_dict=input_feature_dict,
            mask_teacher_forcing=self.teacher_forcing_training.get(batch["gt/valid"], self.current_epoch),
            latent=latent,
            goal=goal_gt,
            goal_valid=goal_valid,
            deterministic_latent=False,
            deterministic_action=self.hparams.training_deterministic_action,
            require_vis_dict=False,
        )

        # ! metrics
        metrics_dict = self.train_metrics_train(
            pred_valid=rollout_buffer.valid,
            diffbar_rewards_valid=rollout_buffer.diffbar_rewards_valid,
            diffbar_rewards=rollout_buffer.diffbar_rewards,
            override_masks=rollout_buffer.override_masks,
            agent_role=batch["ref/agent_role"],
            goal_valid=goal_valid,
            goal_pred=goal_pred,
            goal_gt=goal_gt,
            latent_post=latent_post,
            latent_prior=latent_prior,
        )
        for k in metrics_dict.keys():
            self.log(k, metrics_dict[k], on_step=True)
        self.train_metrics_train.reset()
        return metrics_dict[f"{self.train_metrics_train.prefix}/loss"]

    def reactive_replay(
        self,
        batch: Dict[str, Tensor],
        input_feature_dict: Dict[str, Tensor],
        mask_teacher_forcing: Tensor,
        latent: Optional[MyDist],
        goal: Optional[Tensor],
        goal_valid: Optional[Tensor],
        deterministic_latent: bool,
        deterministic_action: bool,
        require_vis_dict: bool,
    ) -> RolloutBuffer:
        # init traffic rule checker
        rule_checker = TrafficRuleChecker(
            batch["map/boundary"],
            batch["map/valid"],
            batch["map/type"],
            batch["map/pos"],
            batch["map/dir"],
            batch["tl_stop/valid"],
            batch["tl_stop/pos"],
            batch["tl_stop/state"],
            batch["agent/type"],
            batch["agent/size"],
            batch["agent/goal"],
            batch["agent/dest"],
            **self.hparams.traffic_rule_checker,
        )
        # prepare features
        features = {
            "map_valid": input_feature_dict["map_feature_valid"],
            "map_feature": input_feature_dict["map_feature"],
            "tl_valid": input_feature_dict["tl_feature_valid"],
            "tl_feature": input_feature_dict["tl_feature"],
            "agent_type": batch["sc/agent_type"],
            "agent_size": batch["sc/agent_size"],
            # gt states for overriding
            "agent_valid": batch["agent/valid"],
            "vel": batch["agent/vel"],
            "acc": batch["agent/acc"],
            "yaw_rate": batch["agent/yaw_rate"],
            "agent_state": torch.cat([batch["agent/pos"], batch["agent/yaw_bbox"], batch["agent/spd"]], dim=-1),
        }
        rollout_buffer = self.rollout(
            features,
            latent=latent,
            goal=goal,
            goal_valid=goal_valid,
            mask_teacher_forcing=mask_teacher_forcing,
            rule_checker=rule_checker,
            step_start=self.hparams.time_step_sim_start,
            step_end=self.hparams.time_step_end,
            deterministic_latent=deterministic_latent,
            deterministic_action=deterministic_action,
            require_vis_dict=require_vis_dict,
        )
        return rollout_buffer

    def joint_future_pred(
        self,
        batch: Dict[str, Tensor],
        input_feature_dict: Dict[str, Tensor],
        latent: Optional[MyDist],
        goal: Optional[MyDist],
        goal_valid: Optional[Tensor],
        require_vis_dict: bool,
    ) -> Tuple[RolloutBuffer, Tensor, Tensor]:
        k_futures = self.hparams.n_joint_future
        # deterministic tensor for goal and latent: K=0 is deterministic
        deterministic = torch.zeros_like(batch["history/agent/valid"][:, 0])  # [n_batch, n_agent]
        deterministic = deterministic.repeat_interleave(k_futures, 0)  # [n_batch*k_futures, n_agent]
        deterministic[::k_futures] = True

        latent.repeat_interleave_(k_futures, 0)

        rule_checker_agent_dest = None
        rule_checker_agent_goal = None
        goal_sample = None
        if goal is not None:
            goal.repeat_interleave_(k_futures, 0)
            goal_sample = goal.sample(deterministic)
            goal_log_probs = goal.log_prob(goal_sample)
            goal_valid = goal_valid.repeat_interleave(k_futures, 0)

            # traffic rule checker
            if self.model.goal_manager.goal_attr_mode == "dest":
                rule_checker_agent_dest = goal_sample
            elif self.model.goal_manager.goal_attr_mode == "goal_xy":
                rule_checker_agent_goal = goal_sample

        if rule_checker_agent_dest is None and "agent/dest" in batch:
            rule_checker_agent_dest = batch["agent/dest"].repeat_interleave(k_futures, 0)

        if rule_checker_agent_goal is None and "agent/goal" in batch:
            rule_checker_agent_goal = batch["agent/goal"].repeat_interleave(k_futures, 0)

        rule_checker = TrafficRuleChecker(
            batch["map/boundary"].repeat_interleave(k_futures, 0),
            batch["map/valid"].repeat_interleave(k_futures, 0),
            batch["map/type"].repeat_interleave(k_futures, 0),
            batch["map/pos"].repeat_interleave(k_futures, 0),
            batch["map/dir"].repeat_interleave(k_futures, 0),
            batch["history/tl_stop/valid"].repeat_interleave(k_futures, 0),
            batch["history/tl_stop/pos"].repeat_interleave(k_futures, 0),
            batch["history/tl_stop/state"].repeat_interleave(k_futures, 0),
            batch["history/agent/type"].repeat_interleave(k_futures, 0),
            batch["history/agent/size"].repeat_interleave(k_futures, 0),
            rule_checker_agent_goal,  # ground truth goal
            rule_checker_agent_dest,  # pred dest
            **self.hparams.traffic_rule_checker,
        )

        # prepare features
        features = {
            "map_valid": input_feature_dict["map_feature_valid"],
            "map_feature": input_feature_dict["map_feature"],
            "tl_valid": input_feature_dict["tl_feature_valid"],
            "tl_feature": input_feature_dict["tl_feature"],
            "agent_type": batch["sc/agent_type"],
            "agent_size": batch["sc/agent_size"],
            # gt states for overriding
            "agent_valid": batch["agent/valid"],
            "vel": batch["agent/vel"],
            "acc": batch["agent/acc"],
            "yaw_rate": batch["agent/yaw_rate"],
            "agent_state": torch.cat([batch["agent/pos"], batch["agent/yaw_bbox"], batch["agent/spd"]], dim=-1),
        }
        for k in features.keys():
            features[k] = features[k].repeat_interleave(k_futures, 0)
        mask_teacher_forcing = self.teacher_forcing_joint_future_pred.get(features["agent_valid"], self.current_epoch)
        rollout_buffer = self.rollout(
            features,
            latent=latent,
            goal=goal_sample,
            goal_valid=goal_valid,
            mask_teacher_forcing=mask_teacher_forcing,
            rule_checker=rule_checker,
            step_start=self.hparams.time_step_sim_start,
            step_end=self.hparams.time_step_end,
            deterministic_latent=deterministic,
            deterministic_action=True,
            require_vis_dict=require_vis_dict,
        )
        rollout_buffer.flatten_repeat(self.hparams.n_joint_future)
        n_batch, n_agent, n_repeat, _ = rollout_buffer.valid.shape
        goal_log_probs = 0 if goal is None else goal_log_probs.view(n_batch, n_repeat, n_agent).transpose(1, 2)
        if self.model.goal_manager.goal_attr_mode == "dest":
            goal_sample = goal_sample.view(n_batch, n_repeat, n_agent).transpose(1, 2)
        elif self.model.goal_manager.goal_attr_mode == "goal_xy":
            _dim = goal_sample.shape[-1]
            goal_sample = goal_sample.view(n_batch, n_repeat, n_agent, _dim).transpose(1, 2)

        return rollout_buffer, goal_sample, goal_log_probs

    def validation_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        require_vis_dict = batch_idx < self.hparams.n_video_batch
        batch = self.pre_processing(batch)
        input_dict = {k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k}
        latent_post_dict = {k.split("latent_post/")[-1]: v for k, v in batch.items() if "latent_post/" in k}
        latent_prior_dict = {k.split("latent_prior/")[-1]: v for k, v in batch.items() if "latent_prior/" in k}

        input_feature_dict = self.model.encode_input_features(**input_dict)
        latent_post_feature_dict = self.model.encode_input_features(**latent_post_dict)
        latent_prior_feature_dict = self.model.encode_input_features(**latent_prior_dict)

        # ! goal and destination
        goal_gt, goal_valid = self.model.goal_manager.get_gt_goal(
            agent_valid=input_dict["agent_valid"], gt_dest=batch["gt/dest"], gt_goal=batch["gt/goal"]
        )
        goal_pred = self.model.goal_manager.pred_goal(
            agent_type=batch["ref/agent_type"],
            map_type=batch["ref/map_type"],
            agent_state=batch["ref/agent_state"],
            **input_feature_dict,
        )

        # ! latents
        latent_post = self.model.latent_encoder(posterior=True, **latent_post_feature_dict)
        latent_prior = self.model.latent_encoder(**latent_prior_feature_dict)

        # ! reactive_replay: scene reconstruct given a complete episode, same setup as training
        rollout_buffer = self.reactive_replay(
            batch=batch,
            input_feature_dict=input_feature_dict,
            mask_teacher_forcing=self.teacher_forcing_reactive_replay.get(batch["gt/valid"], self.current_epoch),
            latent=latent_post,
            goal=goal_gt,
            goal_valid=goal_valid,
            deterministic_latent=True,
            deterministic_action=True,
            require_vis_dict=require_vis_dict,
        )
        rollout_buffer.flatten_repeat(1)
        self.err_metrics_reactive_replay(
            pred_valid=rollout_buffer.valid,
            pred_states=rollout_buffer.preds,
            gt_valid=batch["gt/valid"][:, self.hparams.time_step_sim_start :].transpose(1, 2),
            gt_states=batch["gt/state"][:, self.hparams.time_step_sim_start :].transpose(1, 2),
            override_masks=rollout_buffer.override_masks,
            agent_role=batch["ref/agent_role"],
        )
        self.rule_metrics_reactive_replay(
            valid=rollout_buffer.valid,
            override_masks=rollout_buffer.override_masks,
            outside_map=rollout_buffer.violations["outside_map"],
            collided=rollout_buffer.violations["collided"],
            run_road_edge=rollout_buffer.violations["run_road_edge"],
            run_red_light=rollout_buffer.violations["run_red_light"],
            passive=rollout_buffer.violations["passive"],
            goal_reached=rollout_buffer.violations["goal_reached"],
            dest_reached=rollout_buffer.violations["dest_reached"],
            agent_type=batch["ref/agent_type"],
        )
        self.train_metrics_reactive_replay(
            pred_valid=rollout_buffer.valid.squeeze(2),
            diffbar_rewards_valid=rollout_buffer.diffbar_rewards_valid.squeeze(2),
            diffbar_rewards=rollout_buffer.diffbar_rewards.squeeze(2),
            override_masks=rollout_buffer.override_masks.squeeze(2),
            agent_role=batch["ref/agent_role"],
            goal_valid=goal_valid,
            goal_pred=goal_pred,
            goal_gt=goal_gt,
            latent_post=latent_post,
            latent_prior=latent_prior,
        )

        # waymo metrics
        pred_dict = self.waymo_post_processing(
            valid=rollout_buffer.valid[:, :, 0].any(-1),
            scores=torch.ones_like(rollout_buffer.preds[:, :, :, 0, 0]),  # [n_batch, n_agent, 1],
            trajs=rollout_buffer.preds[:, :, :, rollout_buffer.step_future_start :],
            agent_type=batch["ref/agent_type"],
        )
        waymo_ops_inputs = self.womd_metrics_reactive_replay(batch, pred_dict["waymo_trajs"], pred_dict["waymo_scores"])
        self.womd_metrics_reactive_replay.aggregate_on_cpu(waymo_ops_inputs)
        self.womd_metrics_reactive_replay.reset()

        self.sub_womd_reactive_replay.add_to_submissions(
            waymo_trajs=pred_dict["waymo_trajs"],  # after nms
            waymo_scores=pred_dict["waymo_scores"],  # after nms
            mask_pred=batch["history/agent/role"][..., 2],
            object_id=batch["history/agent/object_id"],
            scenario_center=batch["scenario_center"],
            scenario_yaw=batch["scenario_yaw"],
            scenario_id=batch["scenario_id"],
        )
        if self.global_rank == 0 and require_vis_dict:
            _pred_goal, _pred_dest = None, None
            if self.model.goal_manager.goal_attr_mode == "dest":
                _pred_dest = goal_pred.sample(True).unsqueeze(2)
            elif self.model.goal_manager.goal_attr_mode == "goal_xy":
                _pred_goal = goal_pred.sample(True).unsqueeze(2)
            self.log_val_video(
                prefix="reactive_replay",
                batch_idx=batch_idx,
                batch=batch,
                buf=rollout_buffer,
                pred_goal=_pred_goal,
                pred_dest=_pred_dest,
                attn_video=False,
            )

        # ! joint_future_pred
        rollout_buffer, goal_sample, goal_log_probs = self.joint_future_pred(
            batch=batch,
            input_feature_dict=input_feature_dict,
            latent=latent_prior,
            goal=goal_pred,
            goal_valid=goal_valid,
            require_vis_dict=require_vis_dict,
        )
        self.err_metrics_joint_future_pred(
            pred_valid=rollout_buffer.valid,
            pred_states=rollout_buffer.preds,
            gt_valid=batch["gt/valid"][:, self.hparams.time_step_sim_start :].transpose(1, 2),
            gt_states=batch["gt/state"][:, self.hparams.time_step_sim_start :].transpose(1, 2),
            override_masks=rollout_buffer.override_masks,
            agent_role=batch["ref/agent_role"],
        )
        self.rule_metrics_joint_future_pred(
            valid=rollout_buffer.valid,
            override_masks=rollout_buffer.override_masks,
            outside_map=rollout_buffer.violations["outside_map"],
            collided=rollout_buffer.violations["collided"],
            run_road_edge=rollout_buffer.violations["run_road_edge"],
            run_red_light=rollout_buffer.violations["run_red_light"],
            passive=rollout_buffer.violations["passive"],
            goal_reached=rollout_buffer.violations["goal_reached"],
            dest_reached=rollout_buffer.violations["dest_reached"],
            agent_type=batch["ref/agent_type"],
        )

        pred_dict = self.waymo_post_processing(
            valid=rollout_buffer.valid[:, :, 0].any(-1),  # [n_scene, n_agent]
            scores=torch.exp(rollout_buffer.latent_log_probs[..., 0] + goal_log_probs),  # [n_batch, n_agent, n_repeat]
            trajs=rollout_buffer.preds[:, :, :, rollout_buffer.step_future_start :],
            agent_type=batch["ref/agent_type"],
        )
        waymo_ops_inputs = self.womd_metrics_joint_future_pred(
            batch, pred_dict["waymo_trajs"], pred_dict["waymo_scores"]
        )
        self.womd_metrics_joint_future_pred.aggregate_on_cpu(waymo_ops_inputs)
        self.womd_metrics_joint_future_pred.reset()

        self.sub_womd_joint_future_pred.add_to_submissions(
            waymo_trajs=pred_dict["waymo_trajs"],  # after nms
            waymo_scores=pred_dict["waymo_scores"],  # after nms
            mask_pred=batch["history/agent/role"][..., 2],
            object_id=batch["history/agent/object_id"],
            scenario_center=batch["scenario_center"],
            scenario_yaw=batch["scenario_yaw"],
            scenario_id=batch["scenario_id"],
        )

        if self.global_rank == 0 and require_vis_dict:
            _pred_goal, _pred_dest = None, None
            if self.model.goal_manager.goal_attr_mode == "dest":
                _pred_dest = goal_sample
            elif self.model.goal_manager.goal_attr_mode == "goal_xy":
                _pred_goal = goal_sample
            # log K=6 preds only for the first batch
            k_to_log = self.hparams.n_joint_future if batch_idx == 0 else 1
            self.log_val_video(
                prefix="joint_future_pred",
                batch_idx=batch_idx,
                batch=batch,
                buf=rollout_buffer,
                pred_scores=pred_dict["waymo_scores"],
                pred_goal=_pred_goal,
                pred_dest=_pred_dest,
                attn_video=False,
                k_to_log=k_to_log,
                as_goal_prior=goal_pred,
            )

    def validation_epoch_end(self, outputs):
        self.log("epoch", self.current_epoch, on_epoch=True)
        epoch_err_metrics_reactive_replay = self.err_metrics_reactive_replay.compute()
        for k, v in epoch_err_metrics_reactive_replay.items():
            self.log(k, v, on_epoch=True)
        self.err_metrics_reactive_replay.reset()
        epoch_rule_metrics_reactive_replay = self.rule_metrics_reactive_replay.compute()
        for k, v in epoch_rule_metrics_reactive_replay.items():
            self.log(k, v, on_epoch=True)
        self.rule_metrics_reactive_replay.reset()
        epoch_train_metrics_reactive_replay = self.train_metrics_reactive_replay.compute()
        for k, v in epoch_train_metrics_reactive_replay.items():
            self.log(k, v, on_epoch=True)
        self.train_metrics_reactive_replay.reset()
        epoch_err_metrics_joint_future_pred = self.err_metrics_joint_future_pred.compute()
        for k, v in epoch_err_metrics_joint_future_pred.items():
            self.log(k, v, on_epoch=True)
        self.err_metrics_joint_future_pred.reset()
        epoch_rule_metrics_joint_future_pred = self.rule_metrics_joint_future_pred.compute()
        for k, v in epoch_rule_metrics_joint_future_pred.items():
            self.log(k, v, on_epoch=True)
        self.rule_metrics_joint_future_pred.reset()

        epoch_womd_metrics_reactive_replay = self.womd_metrics_reactive_replay.compute_waymo_motion_metrics()
        for k, v in epoch_womd_metrics_reactive_replay.items():
            self.log(k, v, on_epoch=True)
        epoch_womd_metrics_joint_future_pred = self.womd_metrics_joint_future_pred.compute_waymo_motion_metrics()
        for k, v in epoch_womd_metrics_joint_future_pred.items():
            self.log(k, v, on_epoch=True)

        self.log(
            "val/loss",
            -epoch_womd_metrics_joint_future_pred[
                f"{self.womd_metrics_joint_future_pred.prefix}/mean_average_precision"
            ],
        )

        if self.global_rank == 0:
            self.sub_womd_reactive_replay.save_sub_files(self.logger[0])
            self.sub_womd_joint_future_pred.save_sub_files(self.logger[0])

    def log_val_video(
        self,
        prefix: str,
        batch_idx: int,
        batch: Dict[str, Tensor],
        buf,
        pred_scores: Optional[Tensor] = None,
        pred_goal: Optional[Tensor] = None,
        pred_dest: Optional[Tensor] = None,
        attn_video: bool = False,
        vis_eps_idx: List[int] = [],
        k_to_log: int = 1,
        as_goal_prior=None,
    ) -> Tuple[List[str], List[str]]:
        n_batch = batch["agent/valid"].shape[0]
        if len(vis_eps_idx) == 0:
            vis_eps_idx = range(n_batch)
        video_paths = []
        image_paths = []
        for idx in vis_eps_idx:
            video_dir = f"video_{batch_idx}-{idx}"
            _path = Path(video_dir)
            _path.mkdir(exist_ok=True, parents=True)
            episode_keys = [
                "agent/valid",
                "agent/pos",
                "agent/yaw_bbox",
                "agent/spd",
                "agent/role",
                "agent/size",
                "agent/goal",
                "map/valid",
                "map/type",
                "map/pos",
                "map/boundary",
                "tl_lane/valid",
                "tl_lane/state",
                "tl_lane/idx",
                "tl_stop/valid",
                "tl_stop/state",
                "tl_stop/pos",
                "tl_stop/dir",
                "episode_idx",
            ]
            episode = {}
            for k in episode_keys:
                episode[k] = batch[k][idx].cpu().numpy()
            # [n_agent, 2]
            episode["agent/dest"] = batch["agent/dest"][idx].cpu().numpy()

            for kf in range(k_to_log):
                # i = idx * k_futures + kf
                # [n_batch, n_step, n_agent, k_pred, 4] x,y,yaw,v -> [n_step, n_agent, 2]
                prediction = {
                    # [n_agent, n_step]
                    "agent/valid": buf.valid[idx, :, kf, buf.step_future_start :],
                    # [n_agent, n_step, 2]
                    "agent/pos": buf.preds[idx, :, kf, buf.step_future_start :, :2],
                    # n_agent, [n_ste,1]
                    "agent/yaw_bbox": buf.preds[idx, :, kf, buf.step_future_start :, [2]],
                    # [n_agent, n_step]
                    "speed": buf.preds[idx, :, kf, buf.step_future_start :, 3],
                    "rew_dr": buf.diffbar_rewards[idx, :, kf, buf.step_future_start :],
                    # [n_agent, n_step]
                    "lat_P": buf.latent_log_probs[idx, :, kf, buf.step_future_start :].float().exp(),
                    "act_P": buf.action_log_probs[idx, :, kf, buf.step_future_start :].float().exp(),
                }
                for k in prediction.keys():
                    prediction[k] = prediction[k].transpose(0, 1).cpu().numpy()
                prediction["step_current"] = self.hparams.time_step_current
                prediction["step_gt"] = self.hparams.time_step_gt
                prediction["step_end"] = self.hparams.time_step_end
                for k in buf.vis_dicts.keys():  # [n_step, n_agent, :]
                    prediction[k] = buf.vis_dicts[k][idx, :, kf, buf.step_future_start :].transpose(0, 1).numpy()
                for k in buf.violations.keys():  # [n_step, n_agent]
                    prediction[k] = buf.violations[k][idx, :, kf, buf.step_future_start :].transpose(0, 1).cpu().numpy()

                if pred_goal is not None:
                    prediction["agent/goal"] = pred_goal[idx, :, kf].cpu().numpy()
                if pred_dest is not None:
                    prediction["agent/dest"] = pred_dest[idx, :, kf].cpu().numpy()
                if pred_scores is not None:
                    prediction["score"] = pred_scores[idx, :, kf].cpu().numpy()

                vis_waymo = VisWaymo(
                    episode["map/valid"], episode["map/type"], episode["map/pos"], episode["map/boundary"]
                )
                video_paths_pred = vis_waymo.save_prediction_videos(f"{video_dir}/{prefix}_K{kf}", episode, prediction)
                video_paths += video_paths_pred
                if attn_video:
                    video_paths_attn = vis_waymo.save_attn_videos(f"{video_dir}/{prefix}_K{kf}", episode, prediction)
                    video_paths += video_paths_attn

                if (as_goal_prior is not None) and (kf == 0):
                    dest_img_paths = vis_waymo.get_dest_prob_image(
                        f"{video_dir}/dest_im", episode, prediction, as_goal_prior.probs[idx].cpu().numpy()
                    )
                    image_paths += dest_img_paths

        if self.logger is not None:
            for v_p in video_paths:
                self.logger[0].experiment.log({v_p: wandb.Video(v_p)}, commit=False)
            for i_p in image_paths:
                self.logger[0].experiment.log({i_p: wandb.Image(i_p)}, commit=False)
        return video_paths, image_paths

    def test_step(self, batch: Dict[str, Tensor], batch_idx: int) -> Dict:
        require_vis_dict = False
        batch = self.pre_processing(batch)
        input_dict = {k.split("input/")[-1]: v for k, v in batch.items() if "input/" in k}
        latent_prior_dict = {k.split("latent_prior/")[-1]: v for k, v in batch.items() if "latent_prior/" in k}
        input_feature_dict = self.model.encode_input_features(**input_dict)
        latent_prior_feature_dict = self.model.encode_input_features(**latent_prior_dict)

        # ! goal and destination
        goal_valid = input_dict["agent_valid"].any(1)
        goal_pred = self.model.goal_manager.pred_goal(
            agent_type=batch["ref/agent_type"],
            map_type=batch["ref/map_type"],
            agent_state=batch["ref/agent_state"],
            **input_feature_dict,
        )

        # ! latents
        latent_prior = self.model.latent_encoder(**latent_prior_feature_dict)

        # ! joint_future_pred
        for k in ["valid", "vel", "acc", "yaw_rate", "pos", "yaw_bbox", "spd", "size"]:
            batch[f"agent/{k}"] = batch[f"history/agent/{k}"]

        rollout_buffer, goal_sample, goal_log_probs = self.joint_future_pred(
            batch=batch,
            input_feature_dict=input_feature_dict,
            latent=latent_prior,
            goal=goal_pred,
            goal_valid=goal_valid,
            require_vis_dict=require_vis_dict,
        )
        pred_dict = self.waymo_post_processing(
            valid=rollout_buffer.valid[:, :, 0].any(-1),  # [n_scene, n_agent]
            scores=torch.exp(rollout_buffer.latent_log_probs[..., 0] + goal_log_probs),  # [n_batch, n_agent, n_repeat]
            trajs=rollout_buffer.preds[:, :, :, rollout_buffer.step_future_start :],
            agent_type=batch["ref/agent_type"],
        )

        self.sub_womd_joint_future_pred.add_to_submissions(
            waymo_trajs=pred_dict["waymo_trajs"],  # after nms
            waymo_scores=pred_dict["waymo_scores"],  # after nms
            mask_pred=batch["history/agent/role"][..., 2],
            object_id=batch["history/agent/object_id"],
            scenario_center=batch["scenario_center"],
            scenario_yaw=batch["scenario_yaw"],
            scenario_id=batch["scenario_id"],
        )

    def test_epoch_end(self, outputs):
        if self.global_rank == 0:
            self.sub_womd_joint_future_pred.save_sub_files(self.logger[0])

    def configure_optimizers(self):
        params = []
        params_goal = []
        for k, v in self.named_parameters():
            if "goal_predictor" in k:
                params_goal.append(v)
            else:
                params.append(v)
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=params)
        if len(params_goal) > 0:
            optimizer.add_param_group({"params": params_goal, "lr": self.hparams.lr_goal})
        scheduler = {
            "scheduler": hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=optimizer),
            "monitor": "val/loss",
            "interval": "epoch",
            "frequency": self.trainer.check_val_every_n_epoch,
            "strict": True,
        }
        return [optimizer], [scheduler]

    def log_grad_norm(self, grad_norm_dict: Dict[str, float]) -> None:
        self.log_dict(grad_norm_dict, on_step=True, on_epoch=False, prog_bar=False, logger=True)
