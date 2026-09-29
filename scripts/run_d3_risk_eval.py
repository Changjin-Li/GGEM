from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.uncertainty.risk_map import RiskMapConfig, build_risk_map, corridor_risk, risk_map_to_features


def ensure_parent(path: str) -> str:
    """Create the parent folder of a relative output path when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC without a scikit-learn dependency."""
    positive = labels.astype(bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(risk: np.ndarray, danger: np.ndarray, n_bins: int = 10) -> float:
    """Measure whether predicted risk matches the observed danger frequency."""
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (risk >= low) & (risk < high)
        if mask.sum() == 0:
            continue
        ece += mask.mean() * abs(danger[mask].mean() - risk[mask].mean())
    return float(ece)


def evaluate(risk_per_frame: np.ndarray, danger: np.ndarray) -> dict:
    """Report discrimination and calibration of the geometric risk signal."""
    return {
        "auroc": auroc(risk_per_frame, danger),
        "ece": expected_calibration_error(risk_per_frame.astype(float), danger.astype(float)),
        "n": int(len(danger)),
        "base_rate": float(danger.mean()),
    }


def self_test(n: int = 2000, seed: int = 0) -> dict:
    """Synthetic check: risk should score high AUROC when correlated with danger."""
    rng = np.random.default_rng(seed)
    risk = rng.random(n)
    noise = rng.normal(0.0, 0.05, n)
    dangerous = (risk + noise > 0.5).astype(float)
    correlated = evaluate(risk, dangerous)
    anti_correlated = evaluate(risk, 1.0 - dangerous)
    return {
        "correlated": correlated,
        "anti_correlated": anti_correlated,
        "ok": correlated["auroc"] > 0.9 and anti_correlated["auroc"] < 0.1,
    }


def risk_map_smoke_test(seed: int = 0) -> dict:
    """Verify the risk map pipeline end to end on synthetic points and a straight trajectory."""
    rng = np.random.default_rng(seed)
    cfg = RiskMapConfig()
    pts = np.concatenate(
        [
            rng.uniform(-40.0, 40.0, (4000, 2)),
            np.full((4000, 1), 0.5),
        ],
        axis=1,
    ).astype(np.float32)
    uncertainty = rng.random(len(pts)).astype(np.float32)
    risk_map = build_risk_map(pts, uncertainty, cfg)
    traj = np.stack([np.linspace(0.0, 35.0, 50), np.zeros(50)], axis=1)
    return {
        "risk_map_shape": list(risk_map.shape),
        "risk_map_max": float(risk_map.max()),
        "features": risk_map_to_features(risk_map, cfg).tolist(),
        "corridor_risk": corridor_risk(risk_map, traj, cfg),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="D3 evaluation: geometric uncertainty as planning risk.")
    parser.add_argument("--risk", help="Per-frame risk scalar, .npy")
    parser.add_argument("--danger", help="Per-frame danger label (0/1), .npy")
    parser.add_argument("--out", default="outputs/d3_report.json")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--risk-map-smoke-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        report = self_test()
    elif args.risk_map_smoke_test:
        report = risk_map_smoke_test()
    elif args.risk and args.danger:
        report = evaluate(np.load(args.risk), np.load(args.danger))
    else:
        raise SystemExit("Provide --risk and --danger, or use --self-test / --risk-map-smoke-test")

    with open(ensure_parent(args.out), "w") as handle:
        json.dump(report, handle, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
