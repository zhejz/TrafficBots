# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional
import torch
from torch import Tensor
from utils.transform_utils import cast_rad
from torch.distributions import kl_divergence, Independent, Normal, OneHotCategoricalStraightThrough


class AngularError:
    def __init__(self, criterion: str, angular_type: Optional[str]) -> None:
        assert angular_type in ["cast", "cosine", "vector", None]
        self.angular_type = angular_type
        if self.angular_type != "cosine":
            self.criterion = getattr(torch.nn, criterion)(reduction="none")

    def compute(self, preds: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            preds: [n_batch, n_step, n_agent, (k_pred), 1]
            target: [n_batch, n_step, n_agent, (k_pred), 1]
        """
        if self.angular_type is None:
            error = self.criterion(preds, target)
        elif self.angular_type == "cast":
            diff_angle = cast_rad(preds - target)
            error = self.criterion(diff_angle, torch.zeros_like(diff_angle))
        elif self.angular_type == "cosine":
            error = 0.5 * (1 - torch.cos(preds - target))
        elif self.angular_type == "vector":
            error_cos = self.criterion(torch.cos(preds), torch.cos(target))
            error_sin = self.criterion(torch.sin(preds), torch.sin(target))
            error = error_cos + error_sin
        return error


class BalancedKL:
    """
    Mastering atari with discrete world models, Algorithm 2: KL Balancing with Automatic Differentiation
    """

    def __init__(self, kl_balance_scale: float, kl_free_nats: float) -> None:
        self.alpha = kl_balance_scale
        self.free_nats = kl_free_nats

    def compute(self, posterior: Normal, prior: Normal) -> Tensor:  # type: ignore
        """
        Args:
            posterior: [n_batch, n_agent, E], Normal
            prior: [n_batch, n_agent, E], Normal
        Return:
            error: [n_batch, n_agent, E]
        """
        if self.alpha > 0:
            # latent dist is either DiagGaussian or MultiCategorical, both are wrapped by Independent
            # assert type(posterior) == Independent
            if type(posterior.base_dist) == OneHotCategoricalStraightThrough:
                detach_post = Independent(OneHotCategoricalStraightThrough(probs=posterior.base_dist.probs.detach()), 1)
                detach_prior = Independent(OneHotCategoricalStraightThrough(probs=prior.base_dist.probs.detach()), 1)
            elif type(posterior.base_dist) == Normal:
                detach_post = Independent(
                    Normal(posterior.base_dist.loc.detach(), posterior.base_dist.scale.detach()), 1
                )
                detach_prior = Independent(Normal(prior.base_dist.loc.detach(), prior.base_dist.scale.detach()), 1)
            error_0 = kl_divergence(detach_post, prior)
            error_1 = kl_divergence(posterior, detach_prior)
            if self.free_nats > 0:
                error_0 = torch.max(error_0, error_0.new_full(error_0.size(), self.free_nats))
                error_1 = torch.max(error_1, error_1.new_full(error_1.size(), self.free_nats))
            error = self.alpha * error_0 + (1 - self.alpha) * error_1
        else:
            error = kl_divergence(posterior, prior)
            if self.free_nats > 0:
                error = torch.max(error, error.new_full(error.size(), self.free_nats))
        return error
