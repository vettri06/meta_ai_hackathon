"""Adaptive curriculum engine using Automatic Domain Randomization (ADR).

Generates custom TASK_CONFIG dicts for the FirewallEnvironment by:
  1. Taking the agent's WeaknessProfile (8-dimension regret vector)
  2. Mapping each weakness dimension to environment parameters
  3. Scaling parameters proportionally to regret (higher regret → harder parameter)
  4. Using the agent's Elo rating to set a global difficulty baseline
  5. Clamping all values to physically valid bounds

The result is a curriculum that:
  - Targets the agent's specific weaknesses (ADR expansion)
  - Maintains moderate challenge on strong dimensions (ADR contraction)
  - Progressively increases global difficulty as the agent improves
  - Never produces degenerate configurations (all params bounded)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from server.self_play.weakness_analyzer import WeaknessProfile, WEAKNESS_DIMENSIONS


# ══════════════════════════════════════════════════════════════════════
# Parameter bounds: [min, max] for each tunable environment parameter
# These define the ADR search space — the curriculum interpolates within
# ══════════════════════════════════════════════════════════════════════

PARAM_BOUNDS: Dict[str, Tuple[float, float]] = {
    "noise_level":          (0.00, 0.15),
    "stealth_multiplier":   (1.0,  2.5),
    "session_ttl_benign":   (2,    5),
    "session_ttl_malicious": (1,   3),
    "escalation_rate_mod":  (0.8,  2.5),
    "false_flag_prob":      (0.00, 0.25),
    "burst_prob":           (0.00, 0.30),
    "burst_size_mult":      (1.5,  4.0),
    "benign_ratio":         (0.55, 0.85),
    "threat_probability":   (0.08, 0.40),
    "traffic_lambda":       (4,    9),
}

# Fixed parameters that don't change with curriculum
FIXED_PARAMS = {
    "max_steps": 500,
    "budget": 300.0,
}


# ══════════════════════════════════════════════════════════════════════
# ADR mapping: weakness dimension → (parameter_name, direction)
#   direction = +1 means higher regret → increase parameter
#   direction = -1 means higher regret → decrease parameter
# ══════════════════════════════════════════════════════════════════════

ADR_MAPPING: Dict[str, List[Tuple[str, float]]] = {
    "detection": [
        ("stealth_multiplier", +1.0),
        ("noise_level", +0.6),
        ("threat_probability", +0.4),
    ],
    "false_positive_ctl": [
        ("false_flag_prob", +1.0),
        ("benign_ratio", +0.5),     # More benign = more FP opportunities
        ("noise_level", +0.3),
    ],
    "stealth_handling": [
        ("stealth_multiplier", +1.0),
        ("escalation_rate_mod", +0.5),
        ("noise_level", +0.4),
    ],
    "false_flag_resist": [
        ("false_flag_prob", +1.0),
        ("noise_level", +0.5),
        ("benign_ratio", +0.3),
    ],
    "burst_resilience": [
        ("burst_prob", +1.0),
        ("burst_size_mult", +0.8),
        ("traffic_lambda", +0.4),
    ],
    "efficiency": [
        ("budget", -0.8),           # Tighter budget
        ("traffic_lambda", +0.5),   # More sessions = more actions needed
        ("burst_prob", +0.3),
    ],
    "early_detection": [
        ("escalation_rate_mod", +1.0),
        ("session_ttl_malicious", -0.6),  # Shorter TTL = less time to detect
        ("stealth_multiplier", +0.3),
    ],
    "cascade_prevention": [
        ("escalation_rate_mod", +1.0),
        ("threat_probability", +0.5),
        ("session_ttl_malicious", -0.4),
    ],
}


@dataclass
class EloState:
    """Elo rating tracker for agent skill level.

    The environment difficulty is treated as a virtual opponent.
    When the agent scores above threshold → it wins → Elo goes up.
    When it scores below → it loses → Elo goes down.
    """
    rating: float = 1000.0
    k_factor: float = 32.0
    history: list = None

    def __post_init__(self):
        if self.history is None:
            self.history = []

    def update(self, agent_score: float, difficulty_rating: float,
               threshold: float = 0.5) -> float:
        """Update Elo rating based on episode outcome.

        Args:
            agent_score: Final grade score from the episode [0, 1]
            difficulty_rating: Virtual Elo rating of the environment config
            threshold: Score above this = win, below = loss

        Returns:
            New Elo rating.
        """
        # Win probability (logistic)
        expected = 1.0 / (1.0 + 10.0 ** ((difficulty_rating - self.rating) / 400.0))

        # Actual outcome: continuous score mapped to [0, 1]
        # Use sigmoid of (score - threshold) for smoother gradients than binary win/loss
        outcome = 1.0 / (1.0 + np.exp(-10.0 * (agent_score - threshold)))

        # Elo update
        delta = self.k_factor * (outcome - expected)
        self.rating = round(self.rating + delta, 2)
        self.history.append({
            "rating": self.rating,
            "delta": round(delta, 2),
            "score": round(agent_score, 4),
            "difficulty": round(difficulty_rating, 2),
            "outcome": round(float(outcome), 4),
        })
        return self.rating

    def to_dict(self) -> Dict:
        return {
            "current_rating": self.rating,
            "total_games": len(self.history),
            "peak_rating": max((h["rating"] for h in self.history), default=self.rating),
            "history": self.history[-10:],  # Last 10 for brevity
        }


class CurriculumEngine:
    """Generates adaptive task configurations using ADR and Elo-based difficulty.

    Core algorithm:
      1. Base difficulty = f(Elo rating) — interpolates between easy/hard bounds
      2. Per-dimension adjustment = regret × ADR weight × direction
      3. Final param = clamp(base + adjustment, param_min, param_max)

    The result is a valid TASK_CONFIG dict that can be passed directly to
    FirewallEnvironment.reset() as a custom configuration.
    """

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.elo = EloState()
        self._generation_count = 0

    def generate(
        self,
        weakness: WeaknessProfile,
        override_elo: float | None = None,
    ) -> Dict:
        """Generate a new task config targeting the agent's weaknesses.

        Args:
            weakness: Current weakness profile from WeaknessAnalyzer
            override_elo: Optional Elo override (uses internal tracker if None)

        Returns:
            Complete TASK_CONFIG dict ready for FirewallEnvironment
        """
        self._generation_count += 1
        elo = override_elo if override_elo is not None else self.elo.rating

        # Step 1: Compute global difficulty fraction from Elo
        # Elo 800 → 0.0 (easiest), Elo 1600 → 1.0 (hardest)
        difficulty_frac = np.clip((elo - 800.0) / 800.0, 0.0, 1.0)

        # Step 2: Compute base parameters via difficulty interpolation
        config = {}
        for param, (lo, hi) in PARAM_BOUNDS.items():
            base = lo + difficulty_frac * (hi - lo)
            config[param] = base

        # Step 3: Apply ADR adjustments from weakness profile
        adjustments: Dict[str, float] = {p: 0.0 for p in PARAM_BOUNDS}

        for dim in WEAKNESS_DIMENSIONS:
            regret = weakness.scores.get(dim, 0.5)
            mappings = ADR_MAPPING.get(dim, [])

            for param, direction in mappings:
                if param not in PARAM_BOUNDS:
                    # Handle 'budget' specially (not in PARAM_BOUNDS, in FIXED_PARAMS)
                    if param == "budget":
                        # Reduce budget when efficiency is weak
                        budget_scale = 1.0 - 0.3 * regret * abs(direction)
                        config["_budget_scale"] = config.get("_budget_scale", 1.0) * budget_scale
                    continue

                lo, hi = PARAM_BOUNDS[param]
                param_range = hi - lo

                # ADR adjustment: proportional to regret and direction
                # regret=0 → no adjustment, regret=1 → full push in direction
                adj = regret * direction * param_range * 0.3
                adjustments[param] += adj

        # Step 4: Apply adjustments and clamp to valid bounds
        for param in PARAM_BOUNDS:
            lo, hi = PARAM_BOUNDS[param]
            config[param] = float(np.clip(
                config[param] + adjustments[param], lo, hi
            ))

        # Step 5: Add fixed parameters
        config["max_steps"] = FIXED_PARAMS["max_steps"]
        budget_base = FIXED_PARAMS["budget"]
        budget_scale = config.pop("_budget_scale", 1.0)
        config["budget"] = round(max(100.0, budget_base * budget_scale), 1)

        # Step 6: Round integer parameters
        config["session_ttl_benign"] = int(round(config["session_ttl_benign"]))
        config["session_ttl_malicious"] = int(round(config["session_ttl_malicious"]))
        config["traffic_lambda"] = int(round(config["traffic_lambda"]))

        # Step 7: Round float parameters for readability
        for param in ["noise_level", "stealth_multiplier", "escalation_rate_mod",
                       "false_flag_prob", "burst_prob", "burst_size_mult",
                       "benign_ratio", "threat_probability"]:
            config[param] = round(config[param], 4)

        # Add metadata
        config["_curriculum_meta"] = {
            "generation": self._generation_count,
            "elo_at_generation": round(elo, 2),
            "difficulty_frac": round(float(difficulty_frac), 4),
            "weakness_worst": weakness.worst_dimension,
            "weakness_mean_regret": round(weakness.mean_regret, 4),
        }

        return config

    def difficulty_rating(self, config: Dict) -> float:
        """Compute a virtual Elo rating for a given config (as opponent strength).

        This is used to feed the Elo update: how hard was this config?
        """
        # Compute normalized difficulty across all parameters
        difficulties = []
        for param, (lo, hi) in PARAM_BOUNDS.items():
            if param in config:
                val = float(config[param])
                frac = (val - lo) / max(hi - lo, 1e-9)
                difficulties.append(np.clip(frac, 0.0, 1.0))

        if not difficulties:
            return 1000.0

        # Map average difficulty [0, 1] → Elo [800, 1600]
        avg_diff = float(np.mean(difficulties))
        return round(800.0 + avg_diff * 800.0, 2)

    def update_elo(self, score: float, config: Dict, threshold: float = 0.5) -> float:
        """Update agent Elo based on episode result.

        Args:
            score: Final episode score [0, 1]
            config: The task config that was played
            threshold: Win/loss threshold

        Returns:
            Updated Elo rating.
        """
        diff_elo = self.difficulty_rating(config)
        return self.elo.update(score, diff_elo, threshold)

    @property
    def current_elo(self) -> float:
        return self.elo.rating

    def to_dict(self) -> Dict:
        return {
            "generation_count": self._generation_count,
            "elo": self.elo.to_dict(),
        }
