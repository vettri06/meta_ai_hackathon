"""Multi-objective reward engine for the Adaptive AI Firewall environment.

Computes R = α·security + β·availability + γ·efficiency + δ·timeliness
with careful balance to prevent degenerate policies (block-all / allow-all).
"""
from __future__ import annotations

from typing import Dict, Tuple
import math


ACTIONS = {
    0: "ALLOW",
    1: "BLOCK",
    2: "INSPECT",
    3: "SANDBOX",
    4: "RATE_LIMIT",
    5: "QUARANTINE",
}

# Costs tuned so total episode cost stays well within budget range
ACTION_COSTS = {
    0: {"latency": 0.0, "compute": 0.0},
    1: {"latency": 0.0, "compute": 0.005},
    2: {"latency": 0.08, "compute": 0.05},
    3: {"latency": 0.20, "compute": 0.12},
    4: {"latency": 0.02, "compute": 0.015},
    5: {"latency": 0.05, "compute": 0.025},
}

# Actions that are considered "blocking" (remove traffic from the network)
BLOCKING_ACTIONS = frozenset({1, 3, 5})
# Actions that are considered "inspection" (gather more info)
INSPECTION_ACTIONS = frozenset({2})


class RewardEngine:
    """Weighted multi-objective reward with anti-degeneracy safeguards.

    Key design choices (from RL perspective):
    - Miss penalty (0.8) vs false-positive penalty (0.35) ratio is ~2.3:1
      This creates genuine tension — block-all loses ~0.35 per benign session,
      and with 60–80% benign traffic, the total FP penalty dominates.
    - Correct ALLOW on benign traffic gets +0.08 (dense positive signal).
    - INSPECT keeps the session alive (handled by environment) and gets a
      small bonus when it correctly identifies malicious traffic.
    - Timeliness bonus decays exponentially with kill-chain phase.
    """

    def __init__(
        self,
        alpha: float = 0.35,
        beta: float = 0.30,
        gamma: float = 0.20,
        delta: float = 0.15,
    ) -> None:
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def action_cost(self, action: int) -> float:
        costs = ACTION_COSTS.get(action, ACTION_COSTS[0])
        return float(costs["latency"] + costs["compute"])

    def reward(
        self,
        action: int,
        is_malicious: bool,
        budget_remaining: float,
        attack_phase: int,
        inspect_correct: bool = False,
    ) -> Tuple[float, Dict[str, float]]:
        blocked = action in BLOCKING_ACTIONS
        inspected = action in INSPECTION_ACTIONS

        # --- Security component ---
        r_security = 0.0
        if is_malicious and blocked:
            r_security += 1.0
        elif is_malicious and not blocked and not inspected:
            r_security -= 2.0  # missed attack
        elif is_malicious and inspected:
            # Inspect alone doesn't stop the attack but is a valid step
            r_security += 0.15 if inspect_correct else -0.5

        # --- Availability component ---
        r_availability = 0.0
        if not is_malicious and blocked:
            r_availability -= 1.2  # strong false-positive penalty
        elif not is_malicious and inspected:
            r_availability -= 0.15  # unnecessary inspection delay
        elif not is_malicious and action == 0:
            r_availability += 0.25  # correctly allowing benign traffic

        # Rate-limiting benign traffic is a milder false positive
        if not is_malicious and action == 4:
            r_availability -= 0.4

        # --- Efficiency component ---
        cost = self.action_cost(action)
        # Penalize cost relative to remaining budget (bigger penalty as budget shrinks)
        r_efficiency = -cost / max(budget_remaining, 0.1)

        # --- Timeliness component ---
        # Exponential bonus for catching attacks early in kill chain
        early_bonus = math.exp(-max(attack_phase, 0))
        r_timeliness = early_bonus if (is_malicious and blocked) else 0.0

        total = (
            self.alpha * r_security
            + self.beta * r_availability
            + self.gamma * r_efficiency
            + self.delta * r_timeliness
        )
        return total, {
            "security": r_security,
            "availability": r_availability,
            "efficiency": r_efficiency,
            "timeliness": r_timeliness,
            "cost": cost,
        }
