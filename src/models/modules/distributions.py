# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Union
from torch import Tensor
import torch
from torch.distributions import Independent, Normal, MultivariateNormal, OneHotCategoricalStraightThrough, Categorical
from torch.nn import functional as F


class MyDist:
    def __init__(self, *args, **kwargs) -> None:
        self.distribution = None

    def log_prob(self, sample: Tensor) -> Tensor:
        """
        log_prob: [n_batch, n_agent]
        """
        return self.distribution.log_prob(sample)

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        """
        Args:
            deterministic: bool, or Tensor for sampling relevant agents and determistic other agents.
        Returns:
            sample: [n_batch, n_agent, out_dim]
        """
        if type(deterministic) is Tensor:
            det_sample = self.distribution.mean
            rnd_sample = self.distribution.rsample()
            sample = det_sample.masked_fill(~deterministic.unsqueeze(-1), 0) + rnd_sample.masked_fill(
                deterministic.unsqueeze(-1), 0
            )
        else:
            if deterministic:
                sample = self.distribution.mean
            else:
                sample = self.distribution.rsample()
        return sample


class DiagGaussian(MyDist):
    def __init__(self, mean: Tensor, log_std: Tensor, valid: Optional[Tensor] = None) -> None:
        """
        mean: [n_batch, n_agent, (k_pred), out_dim]
        covariance_matrix: [n_batch, n_agent, (k_pred), out_dim, out_dim]
        """
        super().__init__()
        self.mean = mean
        self.valid = valid
        self.distribution = Independent(Normal(self.mean, log_std.exp()), 1)
        self.stddev = self.distribution.stddev
        self.covariance_matrix = torch.diag_embed(self.distribution.variance)

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        self.mean = self.mean.repeat_interleave(repeats, dim)
        self.stddev = self.stddev.repeat_interleave(repeats, dim)
        self.distribution = Independent(Normal(self.mean, self.stddev), 1)
        self.covariance_matrix = torch.diag_embed(self.distribution.variance)
        if self.valid is not None:
            self.valid = self.valid.repeat_interleave(repeats, dim)


class Gaussian(MyDist):
    def __init__(self, mean: Tensor, tril: Tensor) -> None:
        """
        mean: [n_batch, n_agent, (k_pred), out_dim]
        tril: [out_dim, out_dim], lower-triangular factor of covariance, with positive-valued diagonal, cov=LL^T
        covariance_matrix: [n_batch, n_agent, (k_pred), out_dim, out_dim]
        """
        super().__init__()
        self.mean = mean
        self.tril = tril
        self.distribution = MultivariateNormal(self.mean, scale_tril=self.tril)
        self.mean = self.distribution.mean
        self.covariance_matrix = self.distribution.covariance_matrix

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        self.mean = self.mean.repeat_interleave(repeats, dim)
        self.tril = self.tril.repeat_interleave(repeats, dim)
        self.distribution = MultivariateNormal(self.mean, scale_tril=self.tril)
        self.mean = self.distribution.mean
        self.covariance_matrix = self.distribution.covariance_matrix


class DummyLatent(MyDist):
    def __init__(self, x, valid) -> None:
        super().__init__()
        self._logp = torch.zeros_like(x[..., 0])
        self._sample = torch.zeros_like(x)
        self.valid = valid

    def log_prob(self, *args, **kwargs) -> Tensor:
        return self._logp

    def sample(self, *args, **kwargs) -> Tensor:
        return self._sample

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        self._logp = self._logp.repeat_interleave(repeats, dim)
        self._sample = self._sample.repeat_interleave(repeats, dim)


class MultiCategorical(MyDist):
    def __init__(self, probs: Tensor, valid: Optional[Tensor] = None):
        """
        probs: [n_batch, n_agent, n_cat, n_class]
        """
        super().__init__()
        self.probs = probs
        self.distribution = Independent(OneHotCategoricalStraightThrough(probs=self.probs), 1)
        self.n_cat = self.probs.shape[-2]
        self.n_class = self.probs.shape[-1]
        self._dtype = self.probs.dtype
        self.valid = valid

    def log_prob(self, sample: Tensor) -> Tensor:
        # [n_batch, n_agent]
        return self.distribution.log_prob(sample.view(*sample.shape[:-1], self.n_cat, self.n_class))

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        """
        Args:
            deterministic: bool, or Tensor for sampling relevant agents and determistic other agents.
        Returns:
            sample: [n_batch, n_agent, out_dim]
        """
        # [n_batch, n_agent, n_cat, n_class]
        if type(deterministic) is Tensor:
            det_sample = (
                F.one_hot(self.distribution.base_dist.probs.argmax(-1), num_classes=self.n_class)
                .type(self._dtype)
                .flatten(start_dim=-2, end_dim=-1)
            )
            rnd_sample = self.distribution.rsample().flatten(start_dim=-2, end_dim=-1)
            sample = det_sample.masked_fill(~deterministic.unsqueeze(-1), 0) + rnd_sample.masked_fill(
                deterministic.unsqueeze(-1), 0
            )
        else:
            if deterministic:
                sample = (
                    F.one_hot(self.distribution.base_dist.probs.argmax(-1), num_classes=self.n_class)
                    .type(self._dtype)
                    .flatten(start_dim=-2, end_dim=-1)
                )
            else:
                sample = self.distribution.rsample().flatten(start_dim=-2, end_dim=-1)
        return sample

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        self.probs = self.probs.repeat_interleave(repeats, dim)
        self.distribution = Independent(OneHotCategoricalStraightThrough(probs=self.probs), 1)
        self.n_cat = self.probs.shape[-2]
        self.n_class = self.probs.shape[-1]
        self._dtype = self.probs.dtype
        if self.valid is not None:
            self.valid = self.valid.repeat_interleave(repeats, dim)


class DestCategorical(MyDist):
    def __init__(self, probs: Optional[Tensor] = None, logits: Optional[Tensor] = None, valid: Optional[Tensor] = None):
        """
        probs: [n_batch, n_agent, n_pl] >= 0, sum up to 1.
        """
        super().__init__()
        if probs is None:
            assert logits is not None
            self.distribution = Categorical(logits=logits)
            self.probs = self.distribution.probs
        else:
            assert probs is not None
            self.distribution = Categorical(probs=probs)
            self.probs = self.distribution.probs

        self.valid = valid

    def log_prob(self, sample: Tensor) -> Tensor:
        # [n_batch, n_agent]
        return self.distribution.log_prob(sample)

    def sample(self, deterministic: Union[bool, Tensor]) -> Tensor:
        """
        Args:
            deterministic: bool, or Tensor for sampling relevant agents and determistic other agents.
        Returns:
            sample: [n_batch, n_agent, out_dim]
        """
        if type(deterministic) is Tensor:
            det_sample = self.distribution.probs.argmax(-1)
            rnd_sample = self.distribution.sample()
            sample = det_sample.masked_fill(~deterministic, 0) + rnd_sample.masked_fill(deterministic, 0)
        else:
            if deterministic:
                sample = self.distribution.probs.argmax(-1)
            else:
                sample = self.distribution.sample()
        return sample

    def repeat_interleave_(self, repeats: int, dim: int) -> None:
        self.probs = self.probs.repeat_interleave(repeats, dim)
        self.distribution = Categorical(probs=self.probs)
        if self.valid is not None:
            self.valid = self.valid.repeat_interleave(repeats, dim)
