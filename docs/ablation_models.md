# Configurations of Ablation Models

Due to the refactoring, we do not have time to reproduce the performance of the ablation models.
Therefore, there might be some bugs and some numbers in the ablation table (Table II of our paper) might differ from the numbers you reproduced.

## Encoder, Investigate PE
- Eq. 1: PE for position, unit vector for direction, concatenate everything and input to MLP (SceneTransformer).
  ```
  model.pre_processing.input.pose_pe.map=pe_xy_unit_dir \
  model.model.input_pe_encoder.pe_mode=input \
  ```
- Eq. 2: PE for position and direction, add PE after the MLP.
  ```
  model.pre_processing.input.pose_pe.map=pe_xy_dir \
  model.pre_processing.input.pe_dim=128 \
  model.model.input_pe_encoder.pe_mode=add \
  ```
## Personality
- Without personality.
  ```
  model.model.latent_encoder.latent_prior.dist_type=dummy \
  model.model.latent_encoder.latent_post.dist_type=dummy \
  model.training_metrics.w_vae_kl=0 \
  ```
- Large KL
  ```
  model.training_metrics.w_vae_kl=1e-2 \
  ```
## Destination
- Without destination.
  ```
  model.model.goal_manager.goal_attr_mode=dummy \
  model.training_metrics.w_goal=0 \
  ```
- Use goal (the polyline associated with the last observed pose) rather than destination.
  ```
  model.model.goal_manager.goal_attr_mode=goal_xy \
  ```
- Use goal without navigator module that drops the goal once it is reached.
  ```
  model.model.goal_manager.goal_attr_mode=goal_xy \
  model.model.goal_manager.disable_if_reached=False \
  ```
## World Model
- Without free nats.
  ```
  model.training_metrics.kl_free_nats=-1 \
  ```
- With action gradients.
  ```
  model.detach_state_policy=False \
  ```
## SimNet
- Without personality & destination.
  ```
  model.model.goal_manager.goal_attr_mode=dummy \
  model.training_metrics.w_goal=0 \
  model.model.latent_encoder.latent_prior.dist_type=dummy \
  model.model.latent_encoder.latent_post.dist_type=dummy \
  model.training_metrics.w_vae_kl=0 \
  ```
- Behavior cloning, i.e. without backpropagation through time.
  ```
  model.teacher_forcing_training.step_horizon=90 \
  ```
- Behavior cloning without personality & destination.
  ```
  model.model.goal_manager.goal_attr_mode=dummy \
  model.training_metrics.w_goal=0 \
  model.model.latent_encoder.latent_prior.dist_type=dummy \
  model.model.latent_encoder.latent_post.dist_type=dummy \
  model.training_metrics.w_vae_kl=0 \
  model.teacher_forcing_training.step_horizon=90 \
  ```
## TrafficSim
- Without bicycle dynamics. Use single integrator.
  ```
  model.dynamics.veh._target_=utils.dynamics.StateIntegrator \
  model.dynamics.cyc._target_=utils.dynamics.StateIntegrator \
  model.dynamics.ped._target_=utils.dynamics.StateIntegrator \
  +model.dynamics.veh.max_v=27 \
  +model.dynamics.cyc.max_v=6 \
  +model.dynamics.ped.max_v=3 \
  ~model.dynamics.veh.max_acc \
  ~model.dynamics.cyc.max_acc \
  ~model.dynamics.ped.max_acc \
  ~model.dynamics.veh.max_yaw_rate \
  ~model.dynamics.cyc.max_yaw_rate \
  ~model.dynamics.ped.max_yaw_rate \
  ~model.dynamics.veh.disable_neg_spd \
  ~model.dynamics.cyc.disable_neg_spd \
  ```
- Interactive decoder.
  ```
  model.model.add_goal_latent_first=True \
  ```
- Resample personality.
  ```
  model.model.resample_latent=True \
  ```