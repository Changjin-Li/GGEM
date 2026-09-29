from __future__ import annotations

import torch
import torch.nn as nn


class RiskAwareSelector(nn.Module):
    """Model-agnostic risk-aware re-scoring for scoring-based planners.

    score' = score - lambda(risk) * risk, where lambda activates selectively:
    the penalty only applies once the risk exceeds a threshold, so that ego
    progress is not needlessly suppressed in safe situations.
    """

    def __init__(self, lam_max: float = 1.0, risk_tau: float = 0.5, temp: float = 0.1):
        super().__init__()
        self.lam_max = lam_max
        self.risk_tau = risk_tau
        self.temp = temp

    def selective_lambda(self, risk: torch.Tensor) -> torch.Tensor:
        return self.lam_max * torch.sigmoid((risk - self.risk_tau) / self.temp)

    def forward(self, base_scores: torch.Tensor, risks: torch.Tensor):
        lam = self.selective_lambda(risks)
        return base_scores - lam * risks, lam


class RiskConditionedRefiner(nn.Module):
    """Regression and diffusion planners cannot be re-scored directly, so the
    risk features predict a conservative refinement of the trajectory instead.
    """

    def __init__(
        self,
        n_steps: int = 8,
        risk_dim: int = 4,
        hidden: int = 128,
        max_delta: float = 1.0,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.max_delta = max_delta
        self.net = nn.Sequential(
            nn.Linear(risk_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, n_steps * 2),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, traj: torch.Tensor, risk_feats: torch.Tensor) -> torch.Tensor:
        delta = self.net(risk_feats).view(-1, self.n_steps, 2)
        return traj + self.max_delta * torch.tanh(delta)


def risk_averse_cost(corridor_risks: torch.Tensor, lam: float = 1.0, margin_m: float = 0.0) -> torch.Tensor:
    """Standalone risk-aware cost for custom planners."""
    return lam * corridor_risks + margin_m


def apply_risk_awareness(planner_type: str, **kwargs):
    """Factory that picks the right risk hook for a given host planner."""
    if planner_type in ("scoring", "drivor", "gtrs-dense"):
        allowed = ("lam_max", "risk_tau", "temp")
        return RiskAwareSelector(**{k: v for k, v in kwargs.items() if k in allowed})
    if planner_type in ("regression", "ltf", "diffusion", "gtrs-dp"):
        allowed = ("n_steps", "risk_dim", "hidden", "max_delta")
        return RiskConditionedRefiner(**{k: v for k, v in kwargs.items() if k in allowed})
    raise ValueError(f"unknown planner_type: {planner_type}")
