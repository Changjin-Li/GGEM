from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass
class ReconResult:
    """Container for a monocular reconstruction.

    Attributes:
        points_world: Dense point cloud in world coordinates, shape (N, 3).
        poses_c2w: Camera-to-world poses, shape (T, 4, 4).
        intrinsics: Camera intrinsics, shape (T, 3, 3).
        colors: Optional per-point RGB, shape (N, 3).
        conf: Optional per-point confidence, shape (N,).
    """

    points_world: np.ndarray
    poses_c2w: np.ndarray
    intrinsics: np.ndarray
    colors: Optional[np.ndarray] = None
    conf: Optional[np.ndarray] = None


class VGGTRunner:
    """Unified wrapper for VGGT short-sequence and VGGT-Long long-sequence reconstruction.

    Args:
        mode: "short" for a single VGGT forward pass, "long" for chunked
            reconstruction with loop closure and global Sim(3) alignment.
        device: Torch device string.
        weights: HuggingFace model id or local checkpoint path.
        chunk: Frames per chunk in "long" mode.
        overlap: Overlapping frames between consecutive chunks.
        loop_closure: Whether to enable loop closure in "long" mode.
    """

    def __init__(
        self,
        mode: str = "long",
        device: str = "cuda",
        weights: str = "facebook/vggt",
        chunk: int = 60,
        overlap: int = 10,
        loop_closure: bool = True,
    ):
        self.mode = mode
        self.device = device
        self.weights = weights
        self.chunk = chunk
        self.overlap = overlap
        self.loop_closure = loop_closure
        self._model = None

    def _load(self):
        if self._model is None:
            from vggt.models.vggt import VGGT
            self._model = VGGT.from_pretrained(self.weights).to(self.device).eval()
        return self._model

    @torch.no_grad()
    def _forward_short(self, images: torch.Tensor) -> dict:
        model = self._load()
        images = images.to(self.device, non_blocking=True)
        return model(images)

    def run(self, images: Sequence[np.ndarray]) -> ReconResult:
        """Reconstruct a temporally ordered list of HxWx3 uint8 images."""
        if self.mode == "short":
            return self._run_single(images)
        return self._run_long(images)

    def _run_single(self, images: Sequence[np.ndarray]) -> ReconResult:
        batch = self._preprocess(images)
        out = self._forward_short(batch)
        return self._pack(out)

    def _run_long(self, images: Sequence[np.ndarray]) -> ReconResult:
        """Chunk, reconstruct each chunk, then align the partial reconstructions."""
        chunks = self._split_chunks(images)
        partials = [self._run_single(chunk) for chunk in chunks]
        return self._align_partials(partials)

    def _split_chunks(self, images: Sequence[np.ndarray]) -> list:
        step = max(1, self.chunk - self.overlap)
        return [list(images[i : i + self.chunk]) for i in range(0, len(images), step)]

    def _preprocess(self, images: Sequence[np.ndarray]) -> torch.Tensor:
        array = np.stack(images).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(0, 3, 1, 2)

    def _pack(self, out: dict) -> ReconResult:
        """Adapt the raw model output. Verify the key names against your VGGT build."""
        points = out["world_points"].reshape(-1, 3).cpu().numpy()
        conf = out.get("world_points_conf")
        conf = conf.reshape(-1).cpu().numpy() if conf is not None else None
        return ReconResult(
            points_world = points.astype(np.float32),
            poses_c2w = out["poses"].cpu().numpy().astype(np.float32),
            intrinsics = out["intrinsics"].cpu().numpy().astype(np.float32),
            conf = conf,
        )

    def _align_partials(self, partials: list) -> ReconResult:
        raise NotImplementedError(
            "Hook up VGGT-Long loop closure and global Sim(3) alignment here. "
            "Prefer GT poses for alignment and use learned registration only for gating."
        )
