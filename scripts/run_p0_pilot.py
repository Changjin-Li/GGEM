from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.vggt.metric_eval import p0_go_no_go
from src.vggt.vggt_runner import VGGTRunner


def ensure_parent(path: str) -> str:
    """Create the parent folder of a relative output path when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def load_images(pattern: str):
    """Load a temporally sorted list of RGB frames from a glob pattern."""
    import imageio.v2 as imageio

    return [imageio.imread(path)[..., :3] for path in sorted(glob.glob(pattern))]


def main() -> None:
    parser = argparse.ArgumentParser(description="P0 pilot: metric accuracy go/no-go.")
    parser.add_argument("--images", required=True, help="Frame image glob or directory.")
    parser.add_argument("--lidar", required=True, help="LiDAR point cloud .npy used as ground truth.")
    parser.add_argument("--gt-poses", required=True, help="Ground-truth trajectory .npy of shape (T, 4, 4).")
    parser.add_argument("--mode", default="long", choices=["short", "long"])
    parser.add_argument("--out", default="outputs/p0_report.json")
    args = parser.parse_args()

    pattern = args.images if "*" in args.images else os.path.join(args.images, "*.jpg")
    images = load_images(pattern)
    print(f"loaded {len(images)} frames")

    recon = VGGTRunner(mode=args.mode).run(images)

    gt_points = np.load(args.lidar).astype(np.float32)
    gt_xyz = np.load(args.gt_poses)[:, :3, 3].astype(np.float32)
    pred_xyz = recon.poses_c2w[:, :3, 3].astype(np.float32)

    report = p0_go_no_go(recon.points_world, gt_points, pred_xyz, gt_xyz)
    with open(ensure_parent(args.out), "w") as handle:
        json.dump(report, handle, indent=2)

    print(json.dumps(report, indent=2))
    print("GO" if report["pass"] else "NO-GO: pivot as planned")


if __name__ == "__main__":
    main()
