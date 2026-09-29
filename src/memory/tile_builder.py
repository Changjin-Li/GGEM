from __future__ import annotations

from dataclasses import dataclass

import numpy as np

BEV_CHANNELS = ["max_height", "mean_height", "log_density", "ground_z", "occupancy"]


@dataclass
class GeoTile:
    """One geometry tile expressed in the lane-local frame.

    Attributes:
        tile_id: Unique tile identifier.
        lane_id: Lane this tile belongs to.
        s_center: Center offset along the lane, in meters.
        length: Tile length along the lane, in meters.
        origin_xy: Tile origin in the lane-local frame.
        theta: Tile heading in the lane-local frame, in radians.
        bev: Multi-channel BEV tensor of shape (len(BEV_CHANNELS), W, W).
        updated_at: Timestamp of the last update, in seconds.
        n_obs: Number of observations fused into this tile.
    """

    tile_id: str
    lane_id: str
    s_center: float
    length: float
    origin_xy: np.ndarray
    theta: float
    bev: np.ndarray
    updated_at: float
    n_obs: int


def lane_local_frame(points: np.ndarray, lane_xy: np.ndarray) -> np.ndarray:
    """Project points into the lane-local frame: x is tangential, y is left-lateral."""
    origin = lane_xy.mean(0)
    tangent = lane_xy[-1] - lane_xy[0]
    tangent = tangent / (np.linalg.norm(tangent) + 1e-9)
    left = np.array([-tangent[1], tangent[0]])
    delta = points[:, :2] - origin
    return np.stack([delta @ tangent, delta @ left], axis=1)


def crop_lane_tiles(
    points: np.ndarray,
    lane_id: str,
    lane_xy: np.ndarray,
    tile_len: float = 20.0,
    overlap: float = 0.5,
    updated_at: float = 0.0,
) -> list:
    """Crop a lane-centered point cloud into overlapping tiles along the lane."""
    local = lane_local_frame(points, lane_xy)
    z = points[:, 2]
    step = tile_len * (1.0 - overlap)
    s_min, s_max = local[:, 0].min(), local[:, 0].max()
    tiles = []
    s = s_min
    while s < s_max:
        mask = (local[:, 0] >= s) & (local[:, 0] < s + tile_len)
        if mask.sum() > 32:
            tiles.append(GeoTile(
                tile_id = f"{lane_id}_{s:.1f}",
                lane_id = lane_id,
                s_center = float(s + tile_len / 2),
                length = tile_len,
                origin_xy = np.array([0.0, 0.0]),
                theta = 0.0,
                bev = encode_bev(local[mask], z[mask], tile_len),
                updated_at = updated_at,
                n_obs = 1,
            ))
        s += step
    return tiles


def encode_bev(local_xy: np.ndarray, z: np.ndarray, tile_len: float, res: float = 0.2) -> np.ndarray:
    """Rasterize tile points into a multi-channel BEV tensor aligned with BEV planners.

    Channels are max height, mean height, log density, ground elevation, and occupancy.
    """
    width = int(round(tile_len / res))
    x_idx = np.clip((local_xy[:, 0] / res).astype(np.int64), 0, width - 1)
    y_idx = np.clip((local_xy[:, 1] / res).astype(np.int64), 0, width - 1)
    flat = x_idx * width + y_idx

    bev = np.zeros((len(BEV_CHANNELS), width * width), np.float32)
    np.maximum.at(bev[0], flat, z)
    np.add.at(bev[1], flat, z)
    np.add.at(bev[2], flat, 1.0)

    occupied = bev[2] > 0
    bev[1][occupied] /= bev[2][occupied]
    bev[2] = np.log1p(bev[2])
    bev[4][occupied] = 1.0
    return bev.reshape(len(BEV_CHANNELS), width, width)
