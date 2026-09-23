"""Gaussian-weighted full-volume inference."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn



def _starts(size: int, window: int, overlap: float) -> list[int]:
    if size <= window:
        return [0]
    stride = max(1, int(round(window * (1.0 - overlap))))
    starts = list(range(0, size - window + 1, stride))
    if starts[-1] != size - window:
        starts.append(size - window)
    return starts


def _gaussian_weight(
    patch_size: Sequence[int], device: torch.device
) -> torch.Tensor:
    axes = []
    for size in patch_size:
        coordinate = torch.linspace(-1.0, 1.0, int(size), device=device)
        axes.append(torch.exp(-0.5 * (coordinate / 0.5).square()))
    weight = (
        axes[0][:, None, None]
        * axes[1][None, :, None]
        * axes[2][None, None, :]
    )
    return weight.clamp_min(1e-3)[None, None]


@torch.inference_mode()
def sliding_window_predict_fields(
    model: nn.Module,
    source: np.ndarray,
    output_keys: Sequence[str],
    patch_size: Sequence[int],
    overlap: float,
    sw_batch_size: int,
    device: torch.device,
    amp_dtype: torch.dtype,
    coordinate_mode: str = "drop_s",
) -> dict[str, np.ndarray]:
    """Overlap-weighted inference for multiple scalar FMP output fields."""

    if source.ndim != 4 or source.shape[0] != 2:
        raise ValueError(f"Expected source [2,D,H,W], got {source.shape}")
    keys = tuple(str(key) for key in output_keys)
    if not keys:
        raise ValueError("At least one FMP output key is required")
    patch = tuple(int(value) for value in patch_size)
    original_shape = tuple(int(value) for value in source.shape[-3:])
    padding = [max(0, requested - actual) for requested, actual in zip(patch, original_shape)]
    source_tensor = torch.from_numpy(source[None]).to(device=device, dtype=torch.float32)
    if any(padding):
        source_tensor = F.pad(
            source_tensor,
            (0, padding[2], 0, padding[1], 0, padding[0]),
            mode="constant",
            value=0.0,
        )
    spatial = source_tensor.shape[-3:]
    coordinates = [
        (d, h, w)
        for d in _starts(spatial[0], patch[0], overlap)
        for h in _starts(spatial[1], patch[1], overlap)
        for w in _starts(spatial[2], patch[2], overlap)
    ]
    accumulation = {
        key: torch.zeros((1, 1, *spatial), device=device, dtype=torch.float32)
        for key in keys
    }
    normalization = torch.zeros((1, 1, *spatial), device=device, dtype=torch.float32)
    weight = _gaussian_weight(patch, device)
    model.eval()
    for start in range(0, len(coordinates), int(sw_batch_size)):
        batch_coordinates = coordinates[start : start + int(sw_batch_size)]
        patches = torch.cat(
            [
                source_tensor[:, :, d : d + patch[0], h : h + patch[1], w : w + patch[2]]
                for d, h, w in batch_coordinates
            ],
            dim=0,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=device.type == "cuda" and amp_dtype != torch.float32,
        ):
            outputs = model(patches, coordinate_mode=coordinate_mode)
        for key in keys:
            value = outputs.get(key)
            if not isinstance(value, torch.Tensor) or value.shape[1:] != (1, *patch):
                raise RuntimeError(
                    f"FMP output {key!r} must be [B,1,{patch}], got "
                    f"{None if not isinstance(value, torch.Tensor) else tuple(value.shape)}"
                )
            for prediction, (d, h, w) in zip(value.float(), batch_coordinates):
                accumulation[key][
                    :, :, d : d + patch[0], h : h + patch[1], w : w + patch[2]
                ] += prediction[None] * weight
        for d, h, w in batch_coordinates:
            normalization[
                :, :, d : d + patch[0], h : h + patch[1], w : w + patch[2]
            ] += weight
    result: dict[str, np.ndarray] = {}
    for key in keys:
        field = (accumulation[key] / normalization.clamp_min(1e-6))[0, 0]
        field = field[: original_shape[0], : original_shape[1], : original_shape[2]]
        result[key] = field.cpu().numpy().astype(np.float32)
    return result
