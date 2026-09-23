"""Shared 3D encoder-decoder blocks."""
from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _groups(channels: int) -> int:
    for candidate in (16, 8, 4, 2, 1):
        if channels % candidate == 0:
            return candidate
    return 1


def _triple(value: int | Sequence[int]) -> tuple[int, int, int]:
    if isinstance(value, int):
        return (value, value, value)
    result = tuple(int(item) for item in value)
    if len(result) != 3:
        raise ValueError(f"Expected three spatial values, got {result}")
    return result


class ResidualBlock3D(nn.Module):
    """Pre-activation residual block with GroupNorm and SiLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | Sequence[int] = 3,
    ) -> None:
        super().__init__()
        kernel = _triple(kernel_size)
        padding = tuple(item // 2 for item in kernel)
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv3d(
            in_channels, out_channels, kernel, padding=padding, bias=False
        )
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel, padding=padding, bias=False
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.skip(values)
        values = self.conv1(F.silu(self.norm1(values)))
        values = self.conv2(F.silu(self.norm2(values)))
        return values + residual


class LargeKernelContext3D(nn.Module):
    """Cheap global-local context at the bottleneck."""

    def __init__(
        self, channels: int, kernel_size: Sequence[int] = (3, 7, 7)
    ) -> None:
        super().__init__()
        kernel = _triple(kernel_size)
        padding = tuple(item // 2 for item in kernel)
        self.norm = nn.GroupNorm(_groups(channels), channels)
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            kernel,
            padding=padding,
            groups=channels,
            bias=False,
        )
        self.expand = nn.Conv3d(channels, 2 * channels, 1)
        self.project = nn.Conv3d(2 * channels, channels, 1)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = values
        values = self.depthwise(F.silu(self.norm(values)))
        values = self.project(F.silu(self.expand(values)))
        return values + residual


class EncoderStage(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        blocks: int,
        kernel_size: Sequence[int],
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for index in range(int(blocks)):
            layers.append(
                ResidualBlock3D(
                    in_channels if index == 0 else out_channels,
                    out_channels,
                    kernel_size,
                )
            )
        self.blocks = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.blocks(values)


class DecoderStage(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv3d(in_channels, out_channels, 1, bias=False)
        self.blocks = nn.Sequential(
            ResidualBlock3D(out_channels + skip_channels, out_channels),
            ResidualBlock3D(out_channels, out_channels),
        )

    def forward(self, values: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        values = F.interpolate(
            values, size=skip.shape[-3:], mode="trilinear", align_corners=False
        )
        values = self.reduce(values)
        return self.blocks(torch.cat([values, skip], dim=1))
