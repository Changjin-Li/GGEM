from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .physical_gate import GateInputs, PhysicalGate


class GeoMemoryModule(nn.Module):
    """Model-agnostic geometry memory injection.

    It edits an intermediate state S of the host planner without changing the
    host architecture and without adding any extra loss. The physical gate
    multiplies the learned gate so that an inconsistent memory is ignored.

    Args:
        state_dim: Channel dimension of the host planner state.
        mem_dim: Output dimension of the frozen geometry encoder.
        n_prior: Number of retrieved prior tiles.
        n_persistent: Number of learnable persistent memory tokens.
        n_heads: Number of attention heads.
    """

    def __init__(
        self,
        state_dim: int,
        mem_dim: int = 256,
        n_prior: int = 20,
        n_persistent: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.n_prior = n_prior
        self.encoder = _FrozenGeoEncoder(mem_dim)
        self.proj_c = nn.Linear(mem_dim, state_dim)
        self.proj_s = nn.Linear(state_dim, state_dim)
        self.persistent = nn.Parameter(torch.randn(n_persistent, state_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(state_dim, n_heads, batch_first=True)
        self.gate_mlp = nn.Sequential(
            nn.Linear(state_dim * 2 + 3, 64), nn.SiLU(), 
            nn.Linear(64, state_dim),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)
        self.phys_gate = PhysicalGate()

    @torch.no_grad()
    def encode_memory(self, tiles, device) -> torch.Tensor:
        """Encode retrieved tiles with the frozen encoder (offline-cacheable)."""
        bev = torch.stack([torch.as_tensor(tile.bev) for tile in tiles]).to(device)
        return self.encoder(bev)

    def forward(
        self,
        state: torch.Tensor,
        mem_tokens: torch.Tensor,
        gate_inputs: Optional[GateInputs] = None,
    ) -> torch.Tensor:
        """Args:
            state: Host planner state of shape (B, state_dim).
            mem_tokens: Encoded geometry tiles of shape (B, K, mem_dim).
            gate_inputs: Verifiable geometric cues; if None the physical gate is bypassed.
        """
        batch = state.shape[0]
        context = self.proj_c(mem_tokens)
        persistent = self.persistent[None].expand(batch, -1, -1)
        memory = torch.cat([persistent, context], dim=1)

        query = self.proj_s(state)[:, None, :]
        retrieved, _ = self.cross_attn(query, memory, memory)
        retrieved = retrieved[:, 0, :]

        cos = torch.cosine_similarity(state, retrieved, dim=-1, eps=1e-6)[:, None]
        dist = torch.norm(state - retrieved, dim=-1, keepdim=True)
        ones = torch.ones(batch, 1, device=state.device)
        gate_learn = torch.sigmoid(self.gate_mlp(torch.cat([state, retrieved, cos, dist, ones], dim=-1)))

        if gate_inputs is not None:
            gate_phys = self.phys_gate(gate_inputs)
        else:
            gate_phys = torch.ones(batch, 1, device=state.device)
        return state + (gate_learn * gate_phys) * retrieved


class _FrozenGeoEncoder(nn.Module):
    """Frozen geometry encoder: precomputed offline so inference stays cheap."""

    def __init__(self, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(5, 32, 5, 2, 2), nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.SiLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(128, out_dim)
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, bev: torch.Tensor) -> torch.Tensor:
        return self.fc(self.net(bev).flatten(1))
