from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import regnet_x_400mf
from torchvision.ops import Conv2dNormActivation, FeaturePyramidNetwork


@dataclass(frozen=True)
class DepthEncoderRuntimeConfig:
    """Depth encoder/noise constants used by the current SRU navigation policy."""

    focal_length: float = 25.0
    baseline: float = 0.12
    min_depth: float = 0.25
    max_depth: float = 10.0
    feature_dim: int = 64
    add_noise: bool = False


class VAESampler(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int):
        super().__init__()
        self.conv = Conv2dNormActivation(input_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.mean_layers = nn.Sequential(
            Conv2dNormActivation(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )
        self.logvar_layers = nn.Sequential(
            Conv2dNormActivation(latent_dim, latent_dim, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mean_layers(self.conv(x))


class EncoderFPN(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        encoder = regnet_x_400mf(weights=None)
        encoder = nn.Sequential(*list(encoder.children())[:-2])
        encoder[0][0] = nn.Conv2d(in_channel, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.enc = encoder[0]
        self.enc_1 = encoder[1][:2]
        self.enc_2 = encoder[1][2]
        self.enc_3 = encoder[1][3]
        self.fpn = FeaturePyramidNetwork([64, 160, 400], out_channel)


class DepthEncoder(EncoderFPN):
    def __init__(self, out_channel: int):
        super().__init__(1, out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            x = x.unsqueeze(1)
        out = OrderedDict()
        x = self.enc(x)
        out["feat1"] = self.enc_1(x)
        out["feat2"] = self.enc_2(out["feat1"])
        out["feat3"] = self.enc_3(out["feat2"])
        return self.fpn(out)["feat1"]


class VAEDecoder(nn.Module):
    def __init__(self, input_dim: int, out_dim: int):
        super().__init__()
        self.conv = Conv2dNormActivation(input_dim, input_dim, kernel_size=3, stride=1, padding=1, bias=False)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(input_dim, input_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(input_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_dim, out_dim, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.conv(z))


class DepthDecoder(VAEDecoder):
    def __init__(self, input_dim: int):
        super().__init__(input_dim, 1)


class VAENet(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        self.depth_encoder = DepthEncoder(latent_dim)
        self.vae_sampler = VAESampler(latent_dim, latent_dim)
        self.depth_decoder = DepthDecoder(latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae_sampler(self.depth_encoder(x))

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.depth_decoder(z)


class DepthNoise(nn.Module):
    def __init__(
        self,
        focal_length: float,
        baseline: float,
        min_depth: float,
        max_depth: float,
        filter_size: int = 3,
        inlier_thred_range: tuple[float, float] = (0.01, 0.05),
        prob_range: tuple[float, float] = (0.4, 0.6),
        invalid_disp: float = 1e7,
    ):
        super().__init__()
        self.focal_length = focal_length
        self.baseline = baseline
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.filter_size = filter_size
        self.inlier_thred_range = inlier_thred_range
        self.prob_range = prob_range
        self.invalid_disp = invalid_disp
        weights, substitutes = self._compute_weights(filter_size)
        self.register_buffer("weights", weights.view(1, 1, filter_size, filter_size))
        self.register_buffer("substitutes", substitutes.view(1, 1, filter_size, filter_size))

    @staticmethod
    def _compute_weights(filter_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        center = filter_size // 2
        idx = torch.arange(filter_size) - center
        x_filter, y_filter = torch.meshgrid(idx, idx, indexing="ij")
        sqr_radius = x_filter**2 + y_filter**2
        sqrt_radius = torch.sqrt(sqr_radius)
        weights = 1 / torch.where(sqr_radius == 0, torch.ones_like(sqrt_radius), sqrt_radius)
        weights = weights / weights.sum()
        fill_weights = 1 / (1 + sqrt_radius)
        fill_weights = torch.where(sqr_radius > filter_size, -1.0, fill_weights)
        substitutes = (fill_weights > 0).float()
        return weights, substitutes

    def filter_disparity(self, disparity: torch.Tensor) -> torch.Tensor:
        b, _, h, w = disparity.shape
        center = self.filter_size // 2
        output = torch.full_like(disparity, self.invalid_disp)
        prob = torch.rand(b, 1, 1, 1, device=disparity.device) * (self.prob_range[1] - self.prob_range[0]) + self.prob_range[0]
        random_mask = torch.rand(b, 1, h, w, device=disparity.device) < prob
        weighted = F.conv2d(disparity, self.weights, padding=center)
        diff = torch.abs(disparity - weighted)
        diff_flat = diff.view(b, -1)
        norm = (diff_flat - torch.mean(diff_flat, dim=1, keepdim=True)) / (torch.std(diff_flat, dim=1, keepdim=True) + 1e-6)
        threshold = torch.rand(b, 1, 1, 1, device=disparity.device) * (self.inlier_thred_range[1] - self.inlier_thred_range[0]) + self.inlier_thred_range[0]
        update_mask = norm.view_as(diff) < threshold
        update_mask = update_mask & random_mask
        disparity = torch.round(disparity * 32.0) / 32.0
        output = torch.where(update_mask, disparity, output)
        filled = F.conv2d(update_mask.float() * disparity, self.substitutes, padding=center)
        counts = F.conv2d(update_mask.float(), self.substitutes, padding=center) + 1e-9
        output = torch.where(counts >= 1, filled / counts, output)
        return output

    def forward(self, depth: torch.Tensor, add_noise: bool) -> torch.Tensor:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        if add_noise:
            depth = torch.clamp(depth, min=1.0 / self.invalid_disp)
            disparity = self.focal_length * self.baseline / depth
            disparity = self.filter_disparity(disparity)
            depth = self.focal_length * self.baseline / disparity
            depth = torch.where(depth < self.min_depth, torch.zeros_like(depth), depth)
        return torch.where(depth > self.max_depth, torch.zeros_like(depth), depth)


class DepthNoiseEncoderRuntime(nn.Module):
    """Lightweight deploy-side copy of the SRU depth noise encoder."""

    def __init__(self, weights_path: str | Path, cfg: DepthEncoderRuntimeConfig | None = None):
        super().__init__()
        self.cfg = cfg or DepthEncoderRuntimeConfig()
        self.depth_noise = DepthNoise(
            focal_length=self.cfg.focal_length,
            baseline=self.cfg.baseline,
            min_depth=self.cfg.min_depth,
            max_depth=self.cfg.max_depth,
        )
        self.encoder = VAENet(self.cfg.feature_dim)
        state = torch.load(Path(weights_path).expanduser(), map_location="cpu", weights_only=True)
        self.encoder.load_state_dict(state, strict=True)

    def forward(self, depth: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if depth.dim() == 3:
            depth = depth.unsqueeze(1)
        noisy_depth = self.depth_noise(depth, add_noise=self.cfg.add_noise)
        encoded_depth = self.encoder(noisy_depth)
        return encoded_depth, noisy_depth
