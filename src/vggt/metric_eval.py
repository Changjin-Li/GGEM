from __future__ import annotations

from typing import Tuple

import numpy as np


def umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    with_scale: bool = True,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """Estimate the optimal Sim(3) mapping src -> dst, so that dst ~= s * R * src + t."""
    assert src.shape == dst.shape and src.shape[1] == 3
    mean_src, mean_dst = src.mean(0), dst.mean(0)
    src_centered, dst_centered = src - mean_src, dst - mean_dst
    cov = dst_centered.T @ src_centered / len(src)
    u, singular, vt = np.linalg.svd(cov)
    fix = np.eye(3)
    fix[2, 2] = np.sign(np.linalg.det(u @ vt))
    rot = u @ fix @ vt
    if with_scale:
        var_src = (src_centered ** 2).sum() / len(src)
        scale = float((singular * np.diag(fix)).sum() / var_src)
    else:
        scale = 1.0
    trans = mean_dst - scale * rot @ mean_src
    return scale, rot.astype(np.float32), trans.astype(np.float32)


def apply_sim3(points: np.ndarray, scale: float, rot: np.ndarray, trans: np.ndarray) -> np.ndarray:
    """Apply an estimated Sim(3) transform to a set of points."""
    return (scale * (rot @ points.T)).T + trans


def ate_rmse(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """Absolute trajectory error RMSE after Sim(3) alignment, in meters."""
    scale, rot, trans = umeyama_sim3(pred_xyz, gt_xyz)
    error = apply_sim3(pred_xyz, scale, rot, trans) - gt_xyz
    return float(np.sqrt((error ** 2).sum(1).mean()))


def scale_drift(pred_xyz: np.ndarray, gt_xyz: np.ndarray) -> float:
    """Global scale error plus segment-wise scale variation, used for the 200 m criterion."""
    scale_global, _, _ = umeyama_sim3(pred_xyz, gt_xyz)
    segment_len = len(pred_xyz) // 4
    segments = [umeyama_sim3(pred_xyz[i : i + segment_len], gt_xyz[i : i + segment_len])[0] for i in range(4)]
    return float(abs(scale_global - 1.0) + (max(segments) - min(segments)) / np.mean(segments))


def chamfer_distance(a: np.ndarray, b: np.ndarray, max_pts: int = 200000) -> float:
    """Symmetric Chamfer distance between two point clouds."""
    a = a[np.random.choice(len(a), min(len(a), max_pts), replace=False)]
    b = b[np.random.choice(len(b), min(len(b), max_pts), replace=False)]
    dist_ab = np.sqrt(((a[:, None, :] - b[None, :, :]) ** 2).sum(-1)).min(1)
    dist_ba = np.sqrt(((b[:, None, :] - a[None, :, :]) ** 2).sum(-1)).min(1)
    return float(0.5 * (dist_ab.mean() + dist_ba.mean()))


def bev_occupancy(points: np.ndarray, res: float = 0.2, size: float = 100.0) -> np.ndarray:
    """Rasterize a point cloud into a square BEV occupancy grid."""
    half = size / 2.0
    mask = (np.abs(points[:, 0]) < half) & (np.abs(points[:, 1]) < half)
    pts = points[mask]
    n = int(size / res)
    idx = np.floor((pts[:, :2] + half) / res).astype(np.int64)
    occupancy = np.zeros((n, n), dtype=np.uint8)
    occupancy[np.clip(idx[:, 0], 0, n - 1), np.clip(idx[:, 1], 0, n - 1)] = 1
    return occupancy


def occupancy_iou(pred_pts: np.ndarray, gt_pts: np.ndarray, res: float = 0.2, size: float = 100.0) -> float:
    """BEV occupancy IoU between a predicted and a ground-truth point cloud."""
    pred = bev_occupancy(pred_pts, res, size)
    gt = bev_occupancy(gt_pts, res, size)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    return float(intersection / max(union, 1))


def p0_go_no_go(
    pred_pts: np.ndarray,
    gt_pts: np.ndarray,
    pred_xyz: np.ndarray,
    gt_xyz: np.ndarray,
    iou_thr: float = 0.70,
    drift_thr: float = 0.05,
) -> dict:
    """Compute the P0 pilot metrics and the go/no-go decision."""
    iou = occupancy_iou(pred_pts, gt_pts)
    drift = scale_drift(pred_xyz, gt_xyz)
    return {
        "bev_iou": iou,
        "scale_drift": drift,
        "ate_rmse": ate_rmse(pred_xyz, gt_xyz),
        "chamfer": chamfer_distance(pred_pts, gt_pts),
        "pass": bool(iou > iou_thr and drift < drift_thr),
    }
