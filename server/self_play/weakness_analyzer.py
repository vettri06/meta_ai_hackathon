"""Regret-based weakness analyzer for the AI Firewall agent.

Analyzes episode statistics across 8 orthogonal capability dimensions
and produces a WeaknessProfile — a normalized regret vector where:
  0.0 = perfect performance (no regret)
  1.0 = worst possible performance (maximum regret)

The profile drives the CurriculumEngine's ADR parameter generation,
focusing training time on dimensions where the agent has the most to gain.

Dimensions:
  1. detection          — Can the agent identify malicious traffic?
  2. false_positive_ctl — Does it avoid blocking benign traffic?
  3. stealth_handling   — Can it catch stealthy/blended attackers?
  4. false_flag_resist  — Does it resist benign-but-suspicious decoys?
  5. burst_resilience   — Can it maintain accuracy during traffic spikes?
  6. efficiency         — Does it manage its action budget wisely?
  7. early_detection    — Does it catch attacks early in the kill chain?
  8. cascade_prevention — Does it prevent late-stage attack escalation?
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


# Dimension names (stable ordering for vector operations)
WEAKNESS_DIMENSIONS = [
    "detection",
    "false_positive_ctl",
    "stealth_handling",
    "false_flag_resist",
    "burst_resilience",
    "efficiency",
    "early_detection",
    "cascade_prevention",
]


@dataclass
class WeaknessProfile:
    """Normalized regret vector across 8 capability dimensions.

    Each value in [0.0, 1.0] where 0.0 = no regret, 1.0 = max regret.
    Higher values indicate dimensions where the agent needs more training.
    """
    scores: Dict[str, float] = field(default_factory=dict)
    raw_metrics: Dict[str, float] = field(default_factory=dict)
    episode_count: int = 0

    def __post_init__(self):
        # Ensure all dimensions are present with default regret = 0.5
        for dim in WEAKNESS_DIMENSIONS:
            self.scores.setdefault(dim, 0.5)

    @property
    def worst_dimension(self) -> str:
        """Dimension with highest regret (most room to improve)."""
        return max(self.scores, key=self.scores.get)

    @property
    def best_dimension(self) -> str:
        """Dimension with lowest regret (strongest capability)."""
        return min(self.scores, key=self.scores.get)

    @property
    def mean_regret(self) -> float:
        """Average regret across all dimensions."""
        return float(np.mean(list(self.scores.values())))

    @property
    def regret_vector(self) -> List[float]:
        """Ordered regret vector for numerical operations."""
        return [self.scores[dim] for dim in WEAKNESS_DIMENSIONS]

    def to_dict(self) -> Dict:
        return {
            "scores": dict(self.scores),
            "raw_metrics": dict(self.raw_metrics),
            "episode_count": self.episode_count,
            "mean_regret": round(self.mean_regret, 4),
            "worst": self.worst_dimension,
            "best": self.best_dimension,
        }


class WeaknessAnalyzer:
    """Analyzes episode statistics into an 8-dimension weakness profile.

    Uses exponential moving average (EMA) across episodes for stability.
    Recent episodes are weighted more heavily (decay factor = alpha).
    """

    def __init__(self, alpha: float = 0.3) -> None:
        """
        Args:
            alpha: EMA smoothing factor. Higher = more reactive to recent episodes.
                   0.3 means ~77% of signal comes from last 3 episodes.
        """
        self.alpha = alpha
        self._running_profile: Optional[WeaknessProfile] = None
        self._episode_history: List[Dict] = []

    def analyze(self, stats: Dict) -> WeaknessProfile:
        """Analyze a single episode's stats and return updated weakness profile.

        Args:
            stats: Output from FirewallEnvironment.get_network_stats()

        Returns:
            Updated WeaknessProfile with EMA-smoothed regret scores.
        """
        # Extract raw metrics from episode stats
        raw = self._extract_raw_metrics(stats)
        self._episode_history.append(raw)

        # Compute per-dimension regret (0 = perfect, 1 = worst)
        instant_scores = self._compute_regret(raw)

        # Apply EMA smoothing
        if self._running_profile is None:
            self._running_profile = WeaknessProfile(
                scores=instant_scores,
                raw_metrics=raw,
                episode_count=1,
            )
        else:
            smoothed = {}
            for dim in WEAKNESS_DIMENSIONS:
                old = self._running_profile.scores[dim]
                new = instant_scores[dim]
                smoothed[dim] = round(self.alpha * new + (1.0 - self.alpha) * old, 6)
            self._running_profile = WeaknessProfile(
                scores=smoothed,
                raw_metrics=raw,
                episode_count=self._running_profile.episode_count + 1,
            )

        return self._running_profile

    def analyze_batch(self, stats_list: List[Dict]) -> WeaknessProfile:
        """Analyze multiple episodes and return final smoothed profile."""
        profile = None
        for stats in stats_list:
            profile = self.analyze(stats)
        return profile or WeaknessProfile()

    def reset(self) -> None:
        """Reset analyzer state for a new training run."""
        self._running_profile = None
        self._episode_history = []

    @property
    def current_profile(self) -> WeaknessProfile:
        return self._running_profile or WeaknessProfile()

    @property
    def history(self) -> List[Dict]:
        return list(self._episode_history)

    # ── Internal ──────────────────────────────────────────────────────

    def _extract_raw_metrics(self, stats: Dict) -> Dict[str, float]:
        """Extract the raw metric values needed for regret computation."""
        return {
            "detection_rate": float(stats.get("detection_rate", 0.0)),
            "false_positive_rate": float(stats.get("false_positive_rate", 0.0)),
            "stealth_detection_rate": float(stats.get("stealth_detection_rate", 0.0)),
            "false_flag_accuracy": float(stats.get("false_flag_accuracy", 0.0)),
            "efficiency": float(stats.get("efficiency", 0.0)),
            "early_detection_bonus": float(stats.get("early_detection_bonus", 0.0)),
            "cascade_prevention": float(stats.get("cascade_prevention", 0.0)),
            "budget_used_pct": float(stats.get("budget_used_pct", 0.0)),
            "burst_ticks": int(stats.get("burst_ticks", 0)),
            "stealth_attacks_seen": int(stats.get("stealth_attacks_seen", 0)),
            "false_flags_seen": int(stats.get("false_flags_seen", 0)),
            "total_malicious": int(stats.get("total_malicious", 0)),
            "total_benign": int(stats.get("total_benign", 0)),
        }

    def _compute_regret(self, raw: Dict[str, float]) -> Dict[str, float]:
        """Compute per-dimension regret from raw metrics.

        Regret is the gap between optimal (0.0) and actual performance (1.0).
        """
        scores = {}

        # 1. Detection: regret = 1 - detection_rate
        scores["detection"] = round(1.0 - raw["detection_rate"], 6)

        # 2. False positive control: regret = false_positive_rate
        #    (0 FP = 0 regret, 100% FP = 1.0 regret)
        scores["false_positive_ctl"] = round(
            min(1.0, raw["false_positive_rate"] * 2.0),  # Scale: 50% FP = max regret
            6,
        )

        # 3. Stealth handling: regret = 1 - stealth_detection_rate
        #    Only meaningful if stealth attacks were seen
        if raw["stealth_attacks_seen"] > 0:
            scores["stealth_handling"] = round(1.0 - raw["stealth_detection_rate"], 6)
        else:
            scores["stealth_handling"] = 0.2  # Mild default regret (untested)

        # 4. False flag resistance: regret = 1 - false_flag_accuracy
        #    Only meaningful if false flags were present
        if raw["false_flags_seen"] > 0:
            scores["false_flag_resist"] = round(1.0 - raw["false_flag_accuracy"], 6)
        else:
            scores["false_flag_resist"] = 0.2  # Mild default regret (untested)

        # 5. Burst resilience: proxy via efficiency during burst-heavy episodes
        #    Higher burst_ticks with maintained efficiency = good
        #    We approximate regret as budget overuse scaled by burst intensity
        burst_factor = min(1.0, raw["burst_ticks"] / 50.0)  # Normalize
        efficiency_gap = 1.0 - raw["efficiency"]
        scores["burst_resilience"] = round(
            min(1.0, efficiency_gap * (1.0 + burst_factor)),
            6,
        )

        # 6. Efficiency: regret = 1 - efficiency
        scores["efficiency"] = round(1.0 - raw["efficiency"], 6)

        # 7. Early detection: regret = 1 - early_detection_bonus
        #    early_detection_bonus is exp(-phase) averaged, range ~[0.05, 1.0]
        scores["early_detection"] = round(
            min(1.0, max(0.0, 1.0 - raw["early_detection_bonus"])),
            6,
        )

        # 8. Cascade prevention: regret = 1 - cascade_prevention
        scores["cascade_prevention"] = round(
            min(1.0, max(0.0, 1.0 - raw["cascade_prevention"])),
            6,
        )

        return scores
