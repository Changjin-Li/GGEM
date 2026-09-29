from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class AppearanceMemoryModule(nn.Module):
    """Appearance-memory control for the E1 gate.

    It mirrors GeoMemoryModule in structure, but the memory is a visual
    embedding and the gate can only come from a learned similarity. It cannot
    compute a registration residual, and that difference is the point of E1.
    """

    def __init__(
        self,
        state_dim: int,
        mem_dim: int = 768,
        n_persistent: int = 4,
        n_heads: int = 4,
    ):
        super().__init__()
        self.proj_c = nn.Linear(mem_dim, state_dim)
        self.proj_s = nn.Linear(state_dim, state_dim)
        self.persistent = nn.Parameter(torch.randn(n_persistent, state_dim) * 0.02)
        self.cross_attn = nn.MultiheadAttention(state_dim, n_heads, batch_first=True)
        self.gate_mlp = nn.Sequential(
            nn.Linear(state_dim * 2 + 3, 64), nn.SiLU(), 
            nn.Linear(64, state_dim),
        )
        nn.init.constant_(self.gate_mlp[-1].bias, -2.0)

    def forward(
        self,
        state: torch.Tensor,
        mem_tokens: torch.Tensor,
        gate_inputs: Optional[object] = None,
    ) -> torch.Tensor:
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
        gate = torch.sigmoid(self.gate_mlp(torch.cat([state, retrieved, cos, dist, ones], dim=-1)))
        return state + gate * retrieved
