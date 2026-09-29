from __future__ import annotations

import numpy as np


def voxel_downsample(points: np.ndarray, voxel: float = 0.2) -> np.ndarray:
    """Keep one representative point per voxel."""
    keys = np.floor(points / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(idx)]


def ground_plane_ransac(
    points: np.ndarray,
    dist_thr: float = 0.25,
    iters: int = 200,
    min_inlier_ratio: float = 0.15,
    seed: int = 0,
):
    """Fit a dominant near-horizontal ground plane with RANSAC.

    Returns (ground_mask, non_ground_mask).
    """
    rng = np.random.default_rng(seed)
    n = len(points)
    best_inliers, best_model = None, None
    for _ in range(iters):
        sel = points[rng.choice(n, 3, replace=False)]
        normal = np.cross(sel[1] - sel[0], sel[2] - sel[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-6:
            continue
        normal = normal / norm
        if normal[2] < 0:
            normal = -normal
        if abs(normal[2]) < 0.8:
            continue
        d = -float(normal @ sel[0])
        inliers = np.abs(points @ normal + d) < dist_thr
        if inliers.mean() < min_inlier_ratio:
            continue
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers, best_model = inliers, (normal, d)
    if best_model is None:
        return np.ones(n, bool), np.zeros(n, bool)
    return best_inliers, ~best_inliers


def visibility_static_mask(
    points: np.ndarray,
    poses_c2w,
    sensor_offset=(0.0, 0.0, 1.8),
    az_bins: int = 720,
    el_bins: int = 120,
    range_margin: float = 1.0,
) -> np.ndarray:
    """Visibility-consistency dynamic removal in the spirit of ERASOR.

    If a frame contains a closer point inside the same (azimuth, elevation) bin,
    the candidate point was not really there at that time and is therefore
    treated as dynamic. Returns a boolean mask where True means kept (static).

    This is the engineering red line for D1: VGGT reconstructs moving vehicles
    as solid geometry, which produces ghost obstacles.
    """
    pts = np.asarray(points, np.float32)
    offset = np.asarray(sensor_offset, np.float32)
    keep = np.ones(len(pts), bool)
    n_bins = az_bins * el_bins
    for pose in poses_c2w:
        inv = np.linalg.inv(np.asarray(pose, np.float32))
        local = (inv[:3, :3] @ pts.T).T + inv[:3, 3] - offset
        radius = np.linalg.norm(local, axis=1)
        valid = radius > 1e-3
        az = np.arctan2(local[:, 1], local[:, 0])
        el = np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1]))
        az_idx = np.clip(((az + np.pi) / (2 * np.pi) * az_bins).astype(np.int64), 0, az_bins - 1)
        el_idx = np.clip(((el + np.pi / 2) / np.pi * el_bins).astype(np.int64), 0, el_bins - 1)
        bin_id = az_idx * el_bins + el_idx
        min_radius = np.full(n_bins, np.inf, np.float32)
        np.minimum.at(min_radius, bin_id[valid], radius[valid])
        shadowed = valid & (radius > min_radius[bin_id] + range_margin)
        keep &= ~shadowed
    return keep


def semantic_dynamic_mask(labels, dynamic_ids=(1, 2, 3, 4, 5, 6, 7, 8)) -> np.ndarray:
    """Keep points whose semantic label is not in the dynamic label set."""
    if labels is None:
        return np.ones(0, bool)
    return ~np.isin(labels, np.asarray(dynamic_ids))


def dynamic_contamination_ratio(points: np.ndarray, poses_c2w, **kwargs) -> float:
    """Gate input: contamination ratio, i.e. the fraction of points judged dynamic."""
    static = visibility_static_mask(np.asarray(points, np.float32), poses_c2w, **kwargs)
    return float(1.0 - static.mean())


def build_static_geometry(
    points: np.ndarray,
    poses_c2w,
    conf=None,
    labels=None,
    voxel: float = 0.2,
    conf_thr=None,
    use_visibility: bool = True,
    use_semantic: bool = True,
) -> np.ndarray:
    """Memory-building entry point: confidence filter, visibility removal,
    semantic removal, then voxel downsampling.
    """
    pts = np.asarray(points, np.float32)
    mask = np.ones(len(pts), bool)
    if conf is not None and conf_thr is not None:
        mask &= np.asarray(conf) >= conf_thr
    if use_visibility and poses_c2w is not None:
        mask &= visibility_static_mask(pts, poses_c2w)
    if use_semantic and labels is not None:
        mask &= semantic_dynamic_mask(labels)
    return voxel_downsample(pts[mask], voxel)
