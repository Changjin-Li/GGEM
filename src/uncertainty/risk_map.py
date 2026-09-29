from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RiskMapConfig:
    """Rasterization settings for the ego-centric epistemic risk map."""

    res: float = 0.2
    size: float = 60.0
    sigma: float = 1.0
    horizon_m: float = 40.0
    topk_frac: float = 0.05


def _kernel1d(sigma: float, res: float) -> np.ndarray:
    radius = int(max(1, round(3.0 * sigma / res)))
    offsets = np.arange(-radius, radius + 1) * res
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _blur2d(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    out = np.apply_along_axis(lambda col: np.convolve(col, kernel, mode="same"), 0, image)
    return np.apply_along_axis(lambda row: np.convolve(row, kernel, mode="same"), 1, out)


def build_risk_map(
    pts_ego: np.ndarray,
    uncertainty: np.ndarray,
    cfg: RiskMapConfig,
) -> np.ndarray:
    """Project per-point geometric uncertainty into an ego-centric BEV risk map."""
    half = cfg.size / 2.0
    n = int(cfg.size / cfg.res)
    pts = np.asarray(pts_ego, np.float32)
    mask = (np.abs(pts[:, 0]) < half) & (np.abs(pts[:, 1]) < half)
    selected_pts = pts[mask]
    selected_unc = np.asarray(uncertainty, np.float32)[mask]
    if len(selected_pts) == 0:
        return np.zeros((n, n), np.float32)
    x_idx = np.clip(np.floor((selected_pts[:, 0] + half) / cfg.res).astype(np.int64), 0, n - 1)
    y_idx = np.clip(np.floor((selected_pts[:, 1] + half) / cfg.res).astype(np.int64), 0, n - 1)
    grid = np.zeros((n, n), np.float32)
    np.add.at(grid, (x_idx, y_idx), selected_unc)
    grid = _blur2d(grid, _kernel1d(cfg.sigma, cfg.res))
    peak = grid.max()
    return grid / peak if peak > 0 else grid


def risk_map_to_features(risk_map: np.ndarray, cfg: RiskMapConfig) -> np.ndarray:
    """Low-dimensional risk features consumed by the gate and the planner."""
    n = risk_map.shape[0]
    half = cfg.size / 2.0
    grid_y, grid_x = np.mgrid[0:n, 0:n]
    x = (grid_x + 0.5) * cfg.res - half
    y = (grid_y + 0.5) * cfg.res - half
    distance = np.hypot(x, y)
    near = risk_map[distance <= 15.0]
    far = risk_map[(distance > 15.0) & (distance <= cfg.horizon_m)]
    return np.array([
        near.max() if near.size else 0.0,
        near.mean() if near.size else 0.0,
        far.max() if far.size else 0.0,
        np.percentile(risk_map, 95),
    ], np.float32)


def corridor_risk(
    risk_map: np.ndarray,
    traj_ego_xy: np.ndarray,
    cfg: RiskMapConfig,
) -> float:
    """Scalar corridor risk: the upper-quantile risk along a candidate trajectory."""
    half = cfg.size / 2.0
    n = risk_map.shape[0]
    traj = np.asarray(traj_ego_xy, np.float32)[:, :2]
    traj = traj[np.linalg.norm(traj, axis=1) <= cfg.horizon_m]
    if len(traj) == 0:
        return 0.0
    x_idx = np.clip(np.floor((traj[:, 0] + half) / cfg.res).astype(np.int64), 0, n - 1)
    y_idx = np.clip(np.floor((traj[:, 1] + half) / cfg.res).astype(np.int64), 0, n - 1)
    values = risk_map[x_idx, y_idx]
    k = max(1, int(np.ceil(cfg.topk_frac * len(values))))
    return float(np.sort(values)[-k:].mean())
