"""Full source-only FMP-Net with exact formation-manifold projection."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .blocks import (
    DecoderStage,
    EncoderStage,
    LargeKernelContext3D,
    ResidualBlock3D,
    _groups,
)
from .formation import (
    FormationManifoldProjector,
    FrozenSWIRenderer,
    attenuation_from_phase_unit,
    phase_unit_from_attenuation,
)


class BottleneckAttention3D(nn.Module):
    """Full order-neutral attention on the compact 3D bottleneck token set."""

    def __init__(
        self,
        channels: int,
        heads: int = 8,
        expansion: float = 2.0,
        layer_scale: float = 1e-3,
    ) -> None:
        super().__init__()
        if channels % int(heads) != 0:
            raise ValueError(f"channels={channels} must be divisible by heads={heads}")
        hidden = int(round(channels * float(expansion)))
        self.norm1 = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(
            channels, int(heads), batch_first=True
        )
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        self.gamma_attention = nn.Parameter(
            torch.full((channels,), float(layer_scale))
        )
        self.gamma_ffn = nn.Parameter(torch.full((channels,), float(layer_scale)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, depth, height, width = values.shape
        tokens = values.flatten(2).transpose(1, 2)
        normalized = self.norm1(tokens)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + self.gamma_attention * attended
        tokens = tokens + self.gamma_ffn * self.ffn(self.norm2(tokens))
        return tokens.transpose(1, 2).reshape(batch, channels, depth, height, width)


class FactorRoleAdapter3D(nn.Module):
    """Lightweight adapter receiving shared and role-specific spectral features."""

    def __init__(self, channels: int, blocks: int = 1) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv3d(2 * channels, channels, 1, bias=False),
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
        ]
        layers.extend(ResidualBlock3D(channels, channels) for _ in range(int(blocks)))
        self.body = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, 1, 1),
        )

    def forward(self, shared: torch.Tensor, role: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(torch.cat([shared, role], dim=1)))


class ScalarHead3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_groups(channels), channels),
            nn.SiLU(),
            nn.Conv3d(channels, 1, 1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.net(values)


class FMPNet3D(nn.Module):
    """Formation-Manifold Projection Network.

    All deployment outputs pass through a frozen renderer.  The provisional SWI
    head supplies a redundant log-coordinate and can never add a post-render
    residual.
    """

    REVISION = "fmp-net-v2-weighted-hard-projection"

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 24,
        channel_multipliers: Sequence[int] = (1, 2, 4, 8, 12),
        blocks_per_stage: int = 2,
        bottleneck_kernel: Sequence[int] = (3, 7, 7),
        attention_heads: int = 8,
        attention_expansion: float = 2.0,
        adapter_blocks: int = 1,
        projection_weights: Sequence[float] = (1.0, 1.0, 4.0),
        frequency_routing: bool = True,
        deep_supervision: bool = True,
        coordinate_minimum: float = -9.0,
        coordinate_maximum: float = math.log(4.0),
        phase_epsilon: float = 1e-4,
    ) -> None:
        super().__init__()
        if int(in_channels) != 2:
            raise ValueError("FMP-Net deployment input must be exactly T1w/T2w")
        channels = [int(base_channels) * int(value) for value in channel_multipliers]
        if len(channels) != 5:
            raise ValueError("FMPNet3D requires five channel levels")
        stem_channels = channels[0] // 2
        if 2 * stem_channels != channels[0]:
            raise ValueError("base_channels must be even")
        self.frequency_routing = bool(frequency_routing)
        self.deep_supervision = bool(deep_supervision)
        self.coordinate_minimum = float(coordinate_minimum)
        self.coordinate_maximum = float(coordinate_maximum)
        self.phase_epsilon = float(phase_epsilon)

        self.t1_stem = nn.Sequential(
            nn.Conv3d(1, stem_channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(_groups(stem_channels), stem_channels),
            nn.SiLU(),
        )
        self.t2_stem = nn.Sequential(
            nn.Conv3d(1, stem_channels, (1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GroupNorm(_groups(stem_channels), stem_channels),
            nn.SiLU(),
        )
        self.enc0 = EncoderStage(channels[0], channels[0], blocks_per_stage, (1, 3, 3))
        self.down1 = nn.Conv3d(
            channels[0], channels[1], (1, 2, 2), stride=(1, 2, 2), bias=False
        )
        self.enc1 = EncoderStage(channels[1], channels[1], blocks_per_stage, (3, 3, 3))
        self.down2 = nn.Conv3d(channels[1], channels[2], 2, stride=2, bias=False)
        self.enc2 = EncoderStage(channels[2], channels[2], blocks_per_stage, (3, 3, 3))
        self.down3 = nn.Conv3d(channels[2], channels[3], 2, stride=2, bias=False)
        self.enc3 = EncoderStage(channels[3], channels[3], blocks_per_stage, (3, 3, 3))
        self.down4 = nn.Conv3d(channels[3], channels[4], 2, stride=2, bias=False)
        self.bottleneck = nn.Sequential(
            EncoderStage(channels[4], channels[4], blocks_per_stage, (3, 3, 3)),
            LargeKernelContext3D(channels[4], bottleneck_kernel),
            BottleneckAttention3D(
                channels[4], attention_heads, attention_expansion
            ),
        )
        self.dec3 = DecoderStage(channels[4], channels[3], channels[3])
        self.dec2 = DecoderStage(channels[3], channels[2], channels[2])
        self.dec1 = DecoderStage(channels[2], channels[1], channels[1])
        self.dec0 = DecoderStage(channels[1], channels[0], channels[0])

        self.magnitude_adapter = FactorRoleAdapter3D(channels[0], adapter_blocks)
        self.phase_adapter = FactorRoleAdapter3D(channels[0], adapter_blocks)
        self.swi_head = ScalarHead3D(channels[0])
        self.aux_magnitude = nn.ModuleList(
            [nn.Conv3d(channels[1], 1, 1), nn.Conv3d(channels[2], 1, 1)]
        )
        self.aux_phase = nn.ModuleList(
            [nn.Conv3d(channels[1], 1, 1), nn.Conv3d(channels[2], 1, 1)]
        )
        self.projector = FormationManifoldProjector(projection_weights)
        self.renderer = FrozenSWIRenderer(phase_epsilon)
        self._initialize_heads()

    def _initialize_heads(self) -> None:
        magnitude_conv = self.magnitude_adapter.head[-1]
        phase_conv = self.phase_adapter.head[-1]
        swi_conv = self.swi_head.net[-1]
        nn.init.zeros_(magnitude_conv.weight)
        nn.init.constant_(magnitude_conv.bias, -0.5)
        nn.init.zeros_(phase_conv.weight)
        nn.init.zeros_(phase_conv.bias)
        nn.init.zeros_(swi_conv.weight)
        nn.init.constant_(swi_conv.bias, math.log(math.expm1(0.7)))
        for head in self.aux_magnitude:
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -0.5)
        for head in self.aux_phase:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def _u(self, raw: torch.Tensor) -> torch.Tensor:
        return raw.clamp(self.coordinate_minimum, self.coordinate_maximum)

    def _q(self, raw: torch.Tensor) -> torch.Tensor:
        return (1.0 - self.phase_epsilon) * torch.tanh(raw)

    def _s(self, raw: torch.Tensor) -> torch.Tensor:
        return (-F.softplus(raw)).clamp(min=self.coordinate_minimum, max=0.0)

    def _spectral_roles(self, feature: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.frequency_routing:
            return feature, feature
        low = F.avg_pool3d(feature, kernel_size=3, stride=1, padding=1)
        high = feature - low
        return low, high

    def _render_projection(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        s: torch.Tensor,
        mode: str,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        projected = dict(self.projector(u, a, s, mode))
        magnitude = torch.exp(projected["u"])
        prediction = self.renderer.from_attenuation(magnitude, projected["a"])
        return projected, prediction

    def forward(
        self,
        values: torch.Tensor,
        coordinate_mode: str = "drop_s",
    ) -> dict[str, Any]:
        if values.ndim != 5 or values.shape[1] != 2:
            raise ValueError(f"Expected [B,2,D,H,W], got {tuple(values.shape)}")
        stem = torch.cat(
            [self.t1_stem(values[:, 0:1]), self.t2_stem(values[:, 1:2])], dim=1
        )
        e0 = self.enc0(stem)
        e1 = self.enc1(self.down1(e0))
        e2 = self.enc2(self.down2(e1))
        e3 = self.enc3(self.down3(e2))
        bottleneck = self.bottleneck(self.down4(e3))
        d3 = self.dec3(bottleneck, e3)
        d2 = self.dec2(d3, e2)
        d1 = self.dec1(d2, e1)
        d0 = self.dec0(d1, e0)

        low, high = self._spectral_roles(d0)
        u_tilde = self._u(self.magnitude_adapter(d0, low))
        q_tilde = self._q(self.phase_adapter(d0, high))
        a_tilde = attenuation_from_phase_unit(q_tilde, self.phase_epsilon)
        s_tilde = self._s(self.swi_head(d0))

        all_projection, all_prediction = self._render_projection(
            u_tilde, a_tilde, s_tilde, "all"
        )
        factor_projection, factor_prediction = self._render_projection(
            u_tilde, a_tilde, s_tilde, "drop_s"
        )
        if coordinate_mode == "all":
            selected_projection, prediction = all_projection, all_prediction
        elif coordinate_mode == "drop_s":
            selected_projection, prediction = factor_projection, factor_prediction
        else:
            selected_projection, prediction = self._render_projection(
                u_tilde, a_tilde, s_tilde, coordinate_mode
            )

        provisional_magnitude = torch.exp(u_tilde)
        unprojected_factor = self.renderer.from_attenuation(
            provisional_magnitude, a_tilde
        )
        auxiliary: tuple[torch.Tensor, ...] = ()

        projected_phase = phase_unit_from_attenuation(
            all_projection["a"], q_tilde
        )
        return {
            "prediction": prediction,
            "auxiliary": auxiliary,
            "coordinate_mode": coordinate_mode,
            "all_prediction": all_prediction,
            "factor_only_prediction": factor_prediction,
            "unprojected_factor_prediction": unprojected_factor,
            "provisional_u": u_tilde,
            "provisional_phase_unit": q_tilde,
            "provisional_attenuation": a_tilde,
            "provisional_s": s_tilde,
            "provisional_magnitude": provisional_magnitude,
            "projected_u": all_projection["u"],
            "projected_attenuation": all_projection["a"],
            "projected_s": all_projection["s"],
            "projected_magnitude": torch.exp(all_projection["u"]),
            "projected_phase_unit": projected_phase,
            "pre_projection_residual": all_projection["pre_residual"],
            "post_projection_residual": all_projection["post_residual"],
            "projection_boundary": all_projection["boundary"],
            "u_correction": all_projection["u_correction"],
            "a_correction": all_projection["a_correction"],
            "s_correction": all_projection["s_correction"],
            "selected_projected_u": selected_projection["u"],
            "selected_projected_attenuation": selected_projection["a"],
            "selected_projected_s": selected_projection["s"],
        }

    def architecture_signature(self) -> dict[str, Any]:
        return {
            "revision": self.REVISION,
            "input_channels": ["T1w", "T2w"],
            "renderer": "raw_neg_n4_d4096",
            "projection_constraint": "s=u-a,a>=0",
            "projection_weights": self.projector.weights.detach().cpu().tolist(),
            "frequency_routing": self.frequency_routing,
            "post_renderer_learned_layers": 0,
        }
