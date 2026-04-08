"""
Diff2Flow Aligner: analytical alignment between diffusion and flow matching trajectories.

Handles:
  - Timestep remapping:  t_FM <-> t_DM
  - Interpolant rescaling: x_FM <-> x_DM
  - Velocity derivation from epsilon prediction

Assumes a variance-preserving (VP) DDPM schedule where:
  alpha_t = sqrt(alpha_bar_t)
  sigma_t = sqrt(1 - alpha_bar_t)
"""

import torch
from diffusers import DDPMScheduler


class Diff2FlowAligner:

    def __init__(
        self,
        noise_scheduler: DDPMScheduler,
        use_timestep_rescaling: bool = True,
        use_interpolant_rescaling: bool = True,
        use_velocity_translation: bool = True,
    ):
        alphas_cumprod = noise_scheduler.alphas_cumprod
        T = len(alphas_cumprod)

        self.alpha = torch.sqrt(alphas_cumprod)
        self.sigma = torch.sqrt(1.0 - alphas_cumprod)
        self.T = T
        self.ft_values = self.alpha / (self.alpha + self.sigma)

        self.use_timestep_rescaling = use_timestep_rescaling
        self.use_interpolant_rescaling = use_interpolant_rescaling
        self.use_velocity_translation = use_velocity_translation

    def t_fm_to_t_dm(self, t_fm: torch.Tensor) -> torch.Tensor:
        """
        Inverse timestep mapping: f_t^{-1}(t_FM) -> t_DM (continuous).

        For each t_FM value, find the two nearest discrete neighbors in ft_values
        and linearly interpolate to get a continuous t_DM.

        When use_timestep_rescaling=False, maps t_FM linearly to [0, T-1]
        without correcting for the DDPM schedule.
        """
        if not self.use_timestep_rescaling:
            return (t_fm * (self.T - 1)).clamp(0, self.T - 1)

        device = t_fm.device
        ft = self.ft_values.to(device)

        ft_ascending = ft.flip(0)
        idx_asc = torch.searchsorted(ft_ascending, t_fm.clamp(ft_ascending[0], ft_ascending[-1]))
        idx_asc = idx_asc.clamp(1, len(ft_ascending) - 1)

        idx_hi_asc = idx_asc
        idx_lo_asc = idx_asc - 1

        ft_lo = ft_ascending[idx_lo_asc]
        ft_hi = ft_ascending[idx_hi_asc]

        t_dm_lo_asc = (self.T - 1 - idx_lo_asc).float()
        t_dm_hi_asc = (self.T - 1 - idx_hi_asc).float()

        denom = (ft_hi - ft_lo).clamp(min=1e-8)
        w = (t_fm - ft_lo) / denom

        t_dm = t_dm_lo_asc + w * (t_dm_hi_asc - t_dm_lo_asc)
        return t_dm

    def get_alpha_sigma(self, t_dm: torch.Tensor) -> tuple:
        """Get interpolated alpha and sigma for continuous t_DM values."""
        device = t_dm.device
        alpha = self.alpha.to(device)
        sigma = self.sigma.to(device)

        t_lo = t_dm.long().clamp(0, self.T - 2)
        t_hi = t_lo + 1
        w = (t_dm - t_lo.float()).clamp(0, 1)

        alpha_t = alpha[t_lo] * (1 - w) + alpha[t_hi] * w
        sigma_t = sigma[t_lo] * (1 - w) + sigma[t_hi] * w

        return alpha_t, sigma_t

    def x_fm_to_x_dm(self, x_fm: torch.Tensor, alpha_t: torch.Tensor, sigma_t: torch.Tensor) -> torch.Tensor:
        """
        f_x^{-1}: transform FM interpolant to DM interpolant.
        x_DM = (alpha + sigma) * x_FM       (Eq. 13)

        When use_interpolant_rescaling=False, returns x_FM unchanged (scale=1).
        """
        if not self.use_interpolant_rescaling:
            return x_fm

        scale = (alpha_t + sigma_t)
        while scale.dim() < x_fm.dim():
            scale = scale.unsqueeze(-1)
        return scale * x_fm

    def eps_to_velocity(
        self,
        eps_pred: torch.Tensor,
        x_dm: torch.Tensor,
        alpha_t: torch.Tensor,
        sigma_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Derive FM velocity from epsilon prediction.

        From epsilon parameterization:
            x0_hat = (x_DM - sigma_t * eps_pred) / alpha_t
            eps_hat = eps_pred
        FM velocity:
            v = x0_hat - eps_hat

        When use_velocity_translation=False, returns eps_pred directly as velocity
        (naive fallback — treats the model's epsilon output as a velocity).
        """
        if not self.use_velocity_translation:
            return eps_pred

        a = alpha_t.clone()
        s = sigma_t.clone()
        while a.dim() < x_dm.dim():
            a = a.unsqueeze(-1)
            s = s.unsqueeze(-1)

        x0_hat = (x_dm - s * eps_pred) / a.clamp(min=1e-8)
        velocity = x0_hat - eps_pred
        return velocity
