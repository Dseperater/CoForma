"""Exact SWI formation coordinates and analytic manifold projection.

The target contract is the train-frozen QTAB operator

    SWI = M * (1 - relu(raw_phase / 4096)) ** 4.

No parameter in this module is learned.  The projection is a closed-form,
voxel-wise weighted Euclidean projection onto ``s = u - a, a >= 0``.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn


def attenuation_from_phase_unit(
    phase_unit: torch.Tensor,
    epsilon: float = 1e-6,
) -> torch.Tensor:
    """Convert signed phase/4096 to the renderer-active log attenuation."""

    active = torch.relu(phase_unit).clamp(max=1.0 - float(epsilon))
    return -4.0 * torch.log((1.0 - active).clamp_min(float(epsilon)))


def phase_unit_from_attenuation(
    attenuation: torch.Tensor,
    provisional_phase_unit: torch.Tensor | None = None,
    epsilon: float = 1e-7,
) -> torch.Tensor:
    """Recover a signed phase coordinate consistent with projected attenuation.

    Positive phase is fixed by attenuation.  If attenuation lies on the inactive
    boundary, the provisional negative phase is retained for signed-phase
    reporting without changing the renderer.
    """

    positive = 1.0 - torch.exp(-torch.clamp_min(attenuation, 0.0) / 4.0)
    if provisional_phase_unit is None:
        return positive
    negative = torch.minimum(provisional_phase_unit, torch.zeros_like(provisional_phase_unit))
    return torch.where(attenuation > float(epsilon), positive, negative)


def render_from_magnitude_attenuation(
    magnitude: torch.Tensor,
    attenuation: torch.Tensor,
    clamp_output: bool = True,
) -> torch.Tensor:
    formed = torch.clamp_min(magnitude, 0.0) * torch.exp(
        -torch.clamp_min(attenuation, 0.0)
    )
    return formed.clamp(0.0, 1.0) if clamp_output else formed


class FrozenSWIRenderer(nn.Module):
    """Immutable differentiable implementation of ``raw_neg_n4_d4096``."""

    def __init__(self, phase_epsilon: float = 1e-6) -> None:
        super().__init__()
        self.phase_epsilon = float(phase_epsilon)

    def forward(
        self,
        magnitude: torch.Tensor,
        phase_unit: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        attenuation = attenuation_from_phase_unit(phase_unit, self.phase_epsilon)
        return render_from_magnitude_attenuation(magnitude, attenuation), attenuation

    def from_attenuation(
        self,
        magnitude: torch.Tensor,
        attenuation: torch.Tensor,
    ) -> torch.Tensor:
        return render_from_magnitude_attenuation(magnitude, attenuation)


def apply_coordinate_omission(
    u: torch.Tensor,
    a: torch.Tensor,
    s: torch.Tensor,
    mode: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Analytically reconstruct one omitted coordinate from the other two."""

    if mode == "all":
        return u, a, s
    if mode == "drop_s":
        return u, a, u - a
    if mode == "drop_u":
        return s + a, a, s
    if mode == "drop_a":
        reconstructed = torch.relu(u - s)
        return u, reconstructed, s
    raise ValueError(f"Unknown formation coordinate mode: {mode}")


class FormationManifoldProjector(nn.Module):
    """Weighted projection onto ``s-u+a=0`` with the boundary ``a>=0``."""

    def __init__(self, weights: Sequence[float] = (1.0, 1.0, 4.0)) -> None:
        super().__init__()
        if len(tuple(weights)) != 3:
            raise ValueError("Projection weights must be (w_u, w_a, w_s)")
        values = torch.as_tensor(tuple(float(item) for item in weights))
        if not bool(torch.all(torch.isfinite(values))) or bool(torch.any(values <= 0)):
            raise ValueError(f"Projection weights must be finite and positive: {weights}")
        self.register_buffer("weights", values.float())

    def forward(
        self,
        u: torch.Tensor,
        a: torch.Tensor,
        s: torch.Tensor,
        mode: str = "all",
    ) -> Mapping[str, torch.Tensor]:
        if u.shape != a.shape or a.shape != s.shape:
            raise ValueError("u, a, and s must have identical shapes")
        u_in, a_in, s_in = apply_coordinate_omission(u, a, s, mode)
        w_u, w_a, w_s = [item.to(dtype=u.dtype, device=u.device) for item in self.weights]
        denominator = w_u.reciprocal() + w_a.reciprocal() + w_s.reciprocal()
        residual = s_in - u_in + a_in
        u_interior = u_in + residual * w_u.reciprocal() / denominator
        a_interior = a_in - residual * w_a.reciprocal() / denominator
        s_interior = s_in - residual * w_s.reciprocal() / denominator

        boundary = a_interior < 0.0
        shared = (w_u * u_in + w_s * s_in) / (w_u + w_s)
        zeros = torch.zeros_like(a_interior)
        u_projected = torch.where(boundary, shared, u_interior)
        a_projected = torch.where(boundary, zeros, a_interior)
        s_projected = torch.where(boundary, shared, s_interior)
        post_residual = s_projected - u_projected + a_projected
        return {
            "u": u_projected,
            "a": a_projected,
            "s": s_projected,
            "input_u": u_in,
            "input_a": a_in,
            "input_s": s_in,
            "pre_residual": residual,
            "post_residual": post_residual,
            "boundary": boundary.to(u.dtype),
            "u_correction": u_projected - u_in,
            "a_correction": a_projected - a_in,
            "s_correction": s_projected - s_in,
        }
