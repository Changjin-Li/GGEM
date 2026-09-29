from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.injection.appearance_baseline import AppearanceMemoryModule
from src.injection.memory_module import GeoMemoryModule
from src.injection.physical_gate import GateInputs

MODALITIES = ("none", "appearance", "geometry", "both")


def ensure_parent(path: str) -> str:
    """Create the parent folder of a relative output path when needed."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    return path


def build_modules(modality: str, state_dim: int):
    """Return (geometry_module, appearance_module) for the requested modality."""
    geo = GeoMemoryModule(state_dim) if modality in ("geometry", "both") else None
    app = AppearanceMemoryModule(state_dim) if modality in ("appearance", "both") else None
    return geo, app


def dummy_gate(batch: int, device) -> GateInputs:
    zeros = torch.zeros(batch, 1, device=device)
    return GateInputs(zeros, torch.ones(batch, 1, device=device), zeros, zeros, zeros)


def dry_run(state_dim: int, batch: int, n_prior: int) -> dict:
    """Check shapes and gate magnitudes for every modality without real data."""
    report = {}
    for modality in MODALITIES:
        geo, app = build_modules(modality, state_dim)
        state = torch.randn(batch, state_dim)
        gate_inputs = dummy_gate(batch, "cpu")
        edited = state
        if geo is not None:
            edited = geo(edited, torch.randn(batch, n_prior, 256), gate_inputs)
        if app is not None:
            edited = app(edited, torch.randn(batch, n_prior, 768), gate_inputs)
        report[modality] = {
            "output_shape": list(edited.shape),
            "mean_delta_norm": float((edited - state).norm(dim=-1).mean()),
        }
    return report


def train_and_eval(modality: str, baseline: str, seed: int, out_dir: str) -> dict:
    """Attach the memory module to a NAVSIM baseline, train jointly, and report EPDMS.

    Note (see plan, experiment E1): run this on weak baselines only
    (LTF 24.7 / GTRS-DP 26.3). The strong baseline DrivoR gains only +1.4,
    which would make the go/no-go experiment uninformative.
    """
    raise NotImplementedError(
        f"NAVSIM training loop not wired yet: modality={modality}, "
        f"baseline={baseline}, seed={seed}, out_dir={out_dir}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="E1 gate: geometry vs appearance memory.")
    parser.add_argument("--baselines", nargs="+", default=["LTF", "GTRS-DP"])
    parser.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--state-dim", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--n-prior", type=int, default=8)
    parser.add_argument("--out", default="outputs/e1_report.json")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(json.dumps(dry_run(args.state_dim, args.batch, args.n_prior), indent=2))
        return

    results = {}
    for baseline in args.baselines:
        for modality in args.modalities:
            for seed in args.seeds:
                key = f"{baseline}|{modality}|seed{seed}"
                results[key] = train_and_eval(modality, baseline, seed, args.out)

    with open(ensure_parent(args.out), "w") as handle:
        json.dump(results, handle, indent=2)

    print("E1 gate: if geometry does not beat appearance, pivot to D3 as planned.")


if __name__ == "__main__":
    main()
