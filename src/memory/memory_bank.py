from __future__ import annotations

import pickle
from collections import defaultdict
from typing import Dict, List

import numpy as np


class MemoryBank:
    """Geometry memory bank: lane indexing, tile storage, and fusion-style updates.

    Args:
        tile_len: Tile length along the lane, in meters.
        overlap: Fractional overlap between neighboring tiles.
    """

    def __init__(self, tile_len: float = 20.0, overlap: float = 0.5):
        self.tile_len = tile_len
        self.overlap = overlap
        self.tiles: Dict[str, object] = {}
        self.lane_graph: Dict[str, List[str]] = defaultdict(list)

    def add_lane(self, lane_id: str, successors: List[str]) -> None:
        """Register the successor lanes of a lane in the connectivity graph."""
        self.lane_graph[lane_id] = list(successors)

    def insert(self, tiles: List[object], t_now: float) -> None:
        """Insert tiles, fusing them into an existing slot when one already exists."""
        for tile in tiles:
            key = round(tile.s_center / (self.tile_len * (1 - self.overlap)))
            gid = f"{tile.lane_id}#{key}"
            if gid in self.tiles:
                self.fuse(gid, tile, t_now)
            else:
                tile.updated_at = t_now
                self.tiles[gid] = tile

    def fuse(self, gid: str, new_tile, t_now: float) -> None:
        """Fusion-style update of an existing tile.

        Metric geometry admits a well-defined running average, whereas an
        appearance embedding has no correct way to be merged, only appended.
        """
        old = self.tiles[gid]
        weight = old.n_obs / (old.n_obs + 1.0)
        old.bev = weight * old.bev + (1.0 - weight) * new_tile.bev
        old.n_obs += 1
        old.updated_at = t_now

    def forward_sequence(self, lane_id: str, intent: str, k: int = 20) -> List[str]:
        """Depth-first forward lane sequence driven by the navigation intent."""
        branch = {"left": 0, "straight": 1, "right": 2}.get(intent, 1)
        sequence, seen = [], set()
        frontier = [lane_id]
        while frontier and len(sequence) < k:
            current = frontier.pop(0)
            if current in seen:
                continue
            seen.add(current)
            sequence.append(current)
            successors = self.lane_graph.get(current, [])
            if successors:
                frontier.append(successors[min(branch, len(successors) - 1)])
        return sequence

    def retrieve(self, lane_id: str, intent: str, s_range: float = 100.0, k: int = 20) -> List[object]:
        """Retrieve up to k tiles along the forward sequence within the lookahead range."""
        sequence = self.forward_sequence(lane_id, intent, k)
        candidates = [tile for tile in self.tiles.values() if tile.lane_id in sequence and abs(tile.s_center) <= s_range]
        candidates.sort(key=lambda tile: tile.s_center)
        return candidates[:k]

    def staleness(self, t_now: float) -> Dict[str, float]:
        """Age of every tile, in seconds, used as a gate input."""
        return {gid: t_now - tile.updated_at for gid, tile in self.tiles.items()}

    def save(self, path: str) -> None:
        """Persist the bank to a relative path."""
        with open(path, "wb") as handle:
            pickle.dump({"tiles": self.tiles, "graph": dict(self.lane_graph)}, handle)

    @classmethod
    def load(cls, path: str) -> "MemoryBank":
        """Restore a bank previously written by save()."""
        with open(path, "rb") as handle:
            payload = pickle.load(handle)
        bank = cls()
        bank.tiles = payload["tiles"]
        bank.lane_graph = defaultdict(list, payload["graph"])
        return bank
