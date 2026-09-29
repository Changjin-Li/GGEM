from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class GateInputs:
    """Quantities that only a metric memory can provide.

    Attributes:
        register_residual: Point-to-plane / ICP residual over the overlap region, in meters.
        overlap_ratio: Visible overlap ratio between the current frame and the memory tile.
        staleness_s: Age of the retrieved memory tile, in seconds.
        dyn_contamination: Fraction of the memory tile judged dynamic, in [0, 1].
        geo_uncertainty: Geometric uncertainty of the retrieved memory, in [0, 1].
    """

    register_residual: torch.Tensor
    overlap_ratio: torch.Tensor
    staleness_s: torch.Tensor
    dyn_contamination: torch.Tensor
    geo_uncertainty: torch.Tensor | None = None


class PhysicalGate(nn.Module):
    """Map verifiable geometric consistency cues to a memory trust gate.

    An appearance prior cannot compute these cues; a metric geometry prior can.
    The gate bias is initialized low (near-closed) so that training starts from
    the plain baseline behaviour.
    """

    def __init__(self, hidden: int = 64, res_tau: float = 0.20, stale_tau: float = 3600.0):
        super().__init__()
        self.res_tau = res_tau
        self.stale_tau = stale_tau
        self.mlp = nn.Sequential(
            nn.Linear(5, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.constant_(self.mlp[-1].bias, -2.0)

    def forward(self, gi: GateInputs) -> torch.Tensor:
        resid = torch.exp(-gi.register_residual / self.res_tau)
        staleness = torch.exp(-gi.staleness_s / self.stale_tau)
        clean = 1.0 - gi.dyn_contamination.clamp(0, 1)
        feas = 1.0 / (1.0 + torch.exp(-gi.register_residual / self.res_tau))
        if gi.geo_uncertainty is None:
            unc = torch.zeros_like(gi.overlap_ratio)
        else:
            unc = gi.geo_uncertainty.clamp(0, 1)
        feats = torch.cat([resid, gi.overlap_ratio, staleness * clean, gi.overlap_ratio * feas, 1.0 - unc], dim=-1)
        return torch.sigmoid(self.mlp(feats))


def point_to_plane_residual(src: torch.Tensor, dst: torch.Tensor, normals: torch.Tensor) -> torch.Tensor:
    """Mean point-to-plane residual in meters over the overlap region."""
    dist = torch.cdist(src, dst)
    nn_idx = dist.argmin(-1)
    diff = src - dst[nn_idx]
    return (diff * normals[nn_idx]).abs().sum(-1).mean()
