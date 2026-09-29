from __future__ import annotations

import numpy as np


def normalize_uncertainty(
    values: np.ndarray,
    lo_pct: float = 5.0,
    hi_pct: float = 95.0,
) -> np.ndarray:
    """Robustly rescale raw uncertainty values into [0, 1]."""
    lo, hi = np.percentile(values, lo_pct), np.percentile(values, hi_pct)
    return np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def local_plane_residual(
    points: np.ndarray,
    voxel: float = 0.5,
    min_pts: int = 10,
) -> np.ndarray:
    """Computable proxy for multi-view consistency.

    Inside each voxel, fit a local plane by PCA and take the point-to-plane
    residual. A large residual means the geometry contradicts itself, which
    indicates a genuinely uncertain reconstruction rather than model noise.
    """
    pts = np.asarray(points, np.float32)
    keys = np.floor(pts / voxel).astype(np.int64)
    _, inverse = np.unique(keys, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    inverse_sorted = inverse[order]
    bounds = np.flatnonzero(np.diff(inverse_sorted)) + 1
    residual = np.zeros(len(pts), np.float32)
    for group in np.split(order, bounds):
        if len(group) < min_pts:
            residual[group] = 1.0
            continue
        group_pts = pts[group]
        center = group_pts.mean(0)
        _, _, vt = np.linalg.svd(group_pts - center, full_matrices=False)
        residual[group] = np.abs((group_pts - center) @ vt[-1])
    return residual


def geometry_uncertainty(
    points: np.ndarray,
    conf=None,
    voxel: float = 0.5,
    w_conf: float = 0.5,
) -> np.ndarray:
    """Combined geometric uncertainty in [0, 1].

    It fuses the VGGT per-point confidence with the multi-view consistency
    residual. The core premise of D3 is that VGGT confidence is physically
    interpretable, since multi-view disagreement means real uncertainty.
    """
    uncertainty_mv = normalize_uncertainty(local_plane_residual(points, voxel))
    if conf is None:
        return uncertainty_mv.astype(np.float32)
    uncertainty_conf = 1.0 - normalize_uncertainty(np.asarray(conf, np.float32))
    return np.clip(w_conf * uncertainty_conf + (1.0 - w_conf) * uncertainty_mv, 0.0, 1.0).astype(np.float32)
