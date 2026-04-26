"""Self-play training arena with mastery gating and Elo progression.

Orchestrates the full self-improvement loop:
  1. Run episode with current curriculum config
  2. Analyze agent weaknesses from episode stats
  3. Update Elo rating based on score vs difficulty
  4. Generate next curriculum targeting identified weaknesses
  5. Apply mastery gate: advance base difficulty only after N consecutive passes

The arena supports any policy function with signature:
  policy(env: FirewallEnvironment, session_ids: List[str]) -> Dict[str, int]

This means it works with both:
  - The heuristic agent (for offline curriculum testing)
  - An LLM-backed agent (for online self-play training)
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from server.firewall_environment import FirewallEnvironment, TASK_CONFIGS
from server.graders import grade_stats
from server.self_play.weakness_analyzer import WeaknessAnalyzer, WeaknessProfile
from server.self_play.curriculum_engine import CurriculumEngine


@dataclass
class RoundResult:
    """Result of a single self-play training round."""
    round_num: int
    score: float
    passed: bool
    elo: float
    elo_delta: float
    difficulty_elo: float
    weakness_profile: Dict[str, float]
    worst_weakness: str
    config_summary: Dict[str, Any]
    stats_summary: Dict[str, float]
    elapsed_seconds: float
    mastery_streak: int

    def to_dict(self) -> Dict:
        return {
            "round": self.round_num,
            "score": round(self.score, 4),
            "passed": self.passed,
            "elo": round(self.elo, 2),
            "elo_delta": round(self.elo_delta, 2),
            "difficulty_elo": round(self.difficulty_elo, 2),
            "weakness": self.weakness_profile,
            "worst_weakness": self.worst_weakness,
            "config": self.config_summary,
            "stats": self.stats_summary,
            "elapsed": round(self.elapsed_seconds, 2),
            "mastery_streak": self.mastery_streak,
        }


class SelfPlayArena:
    """Self-play training arena with adaptive curriculum and Elo progression.

    Usage:
        arena = SelfPlayArena(seed=42)
        results = arena.train(policy=heuristic_policy, num_rounds=30)
        arena.save_history("training_results.json")
    """

    def __init__(
        self,
        seed: int = 42,
        mastery_window: int = 3,
        pass_threshold: float = 0.55,
        ema_alpha: float = 0.3,
    ) -> None:
        """
        Args:
            seed: Random seed for reproducibility
            mastery_window: Consecutive passes needed before difficulty advances
            pass_threshold: Score threshold for a "pass" (used in mastery gating + Elo)
            ema_alpha: EMA smoothing factor for weakness analysis
        """
        self.seed = seed
        self.mastery_window = mastery_window
        self.pass_threshold = pass_threshold

        self.analyzer = WeaknessAnalyzer(alpha=ema_alpha)
        self.curriculum = CurriculumEngine(seed=seed)

        self._results: List[RoundResult] = []
        self._mastery_streak = 0
        self._difficulty_advances = 0
        self._rng = np.random.default_rng(seed)

    def train(
        self,
        policy: Callable,
        num_rounds: int = 30,
        verbose: bool = True,
        callback: Optional[Callable[[RoundResult], None]] = None,
    ) -> List[RoundResult]:
        """Run the full self-play training loop.

        Args:
            policy: Policy function: (env, session_ids) -> {sid: action}
            num_rounds: Number of training rounds
            verbose: Print per-round progress
            callback: Optional callback invoked after each round

        Returns:
            List of RoundResult objects for analysis.
        """
        if verbose:
            self._print_header()

        for round_num in range(1, num_rounds + 1):
            t0 = time.time()

            # 1. Generate curriculum config
            config = self.curriculum.generate(self.analyzer.current_profile)
            config_meta = config.pop("_curriculum_meta", {})

            # 2. Run episode with this config
            env = FirewallEnvironment(seed=self.seed + round_num)
            score, stats = self._run_episode(env, policy, config)

            # 3. Analyze weaknesses
            weakness = self.analyzer.analyze(stats)

            # 4. Update Elo
            elo_before = self.curriculum.current_elo
            self.curriculum.update_elo(score, config, threshold=self.pass_threshold)
            elo_after = self.curriculum.current_elo
            elo_delta = elo_after - elo_before

            # 5. Mastery gating
            passed = score >= self.pass_threshold
            if passed:
                self._mastery_streak += 1
            else:
                self._mastery_streak = 0

            if self._mastery_streak >= self.mastery_window:
                self._difficulty_advances += 1
                self._mastery_streak = 0  # Reset streak after advance

            # 6. Record result
            elapsed = time.time() - t0
            difficulty_elo = self.curriculum.difficulty_rating(config)

            result = RoundResult(
                round_num=round_num,
                score=score,
                passed=passed,
                elo=elo_after,
                elo_delta=elo_delta,
                difficulty_elo=difficulty_elo,
                weakness_profile={k: round(v, 3) for k, v in weakness.scores.items()},
                worst_weakness=weakness.worst_dimension,
                config_summary={
                    "noise": round(config.get("noise_level", 0), 3),
                    "stealth": round(config.get("stealth_multiplier", 1), 2),
                    "ff_prob": round(config.get("false_flag_prob", 0), 3),
                    "burst": round(config.get("burst_prob", 0), 3),
                    "escal": round(config.get("escalation_rate_mod", 1), 2),
                    "budget": round(config.get("budget", 300), 1),
                },
                stats_summary={
                    "det": round(stats.get("detection_rate", 0), 3),
                    "fp": round(stats.get("false_positive_rate", 0), 3),
                    "eff": round(stats.get("efficiency", 0), 3),
                    "stealth_det": round(stats.get("stealth_detection_rate", 0), 3),
                    "ff_acc": round(stats.get("false_flag_accuracy", 0), 3),
                },
                elapsed_seconds=elapsed,
                mastery_streak=self._mastery_streak,
            )

            self._results.append(result)

            if verbose:
                self._print_round(result)

            if callback:
                callback(result)

        if verbose:
            self._print_summary()

        return self._results

    @property
    def results(self) -> List[RoundResult]:
        return list(self._results)

    @property
    def elo_history(self) -> List[float]:
        return [r.elo for r in self._results]

    @property
    def score_history(self) -> List[float]:
        return [r.score for r in self._results]

    def save_history(self, path: str | Path) -> None:
        """Save full training history to JSON."""
        data = {
            "config": {
                "seed": self.seed,
                "mastery_window": self.mastery_window,
                "pass_threshold": self.pass_threshold,
                "num_rounds": len(self._results),
            },
            "summary": self._compute_summary(),
            "rounds": [r.to_dict() for r in self._results],
            "elo": self.curriculum.to_dict(),
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    # ══════════════════════════════════════════════════════════════════
    # Internal
    # ══════════════════════════════════════════════════════════════════

    def _run_episode(
        self,
        env: FirewallEnvironment,
        policy: Callable,
        config: Dict,
    ) -> Tuple[float, Dict]:
        """Run a full episode with a custom config and return (score, stats).

        Injects the generated curriculum config into TASK_CONFIGS temporarily.
        Uses the multi-session step loop but grades externally to avoid
        the grader needing a TASK_SPECS entry for 'curriculum'.
        """
        # Use 'medium' as the base task key — we override its config temporarily
        task_key = "medium"
        original_config = TASK_CONFIGS[task_key].copy()

        try:
            # Temporarily replace medium's config with our curriculum config
            TASK_CONFIGS[task_key] = config
            env.reset(task=task_key, seed=self.seed + self._rng.integers(0, 10000))

            done = False
            step_count = 0
            max_steps = config.get("max_steps", 500)

            while not done and step_count < max_steps:
                session_ids = (
                    list(env.inspected_sessions.keys())
                    + list(env.pending_sessions.keys())
                )
                if not session_ids:
                    # No sessions to act on — advance tick
                    env.current_tick += 1
                    env._expire_sessions()
                    env._spawn_sessions()
                    env._rebuild_queue()
                    step_count += 1
                    continue

                actions = policy(env, session_ids)

                # Apply actions directly (avoids env.step() which calls grade_stats)
                step_reward = 0.0
                for session_id, action in actions.items():
                    if session_id in env.pending_sessions or session_id in env.inspected_sessions:
                        reward, _ = env._apply_action(session_id, action)
                        step_reward += reward

                expired_penalty = env._expire_sessions()
                step_reward += expired_penalty
                env.total_reward += step_reward
                env.step_count += 1
                env.current_tick += 1
                step_count += 1

                done = env.step_count >= max_steps or env.budget_remaining <= 0.0

                if not done:
                    env._spawn_sessions()
                    env._rebuild_queue()

            stats = env.get_network_stats()

            # Grade using medium-task weights (balanced across all metrics)
            grade = grade_stats("medium", stats)
            return float(grade["score"]), stats

        finally:
            # Restore original medium config
            TASK_CONFIGS[task_key] = original_config

    def _compute_summary(self) -> Dict:
        """Compute training summary statistics."""
        if not self._results:
            return {}

        scores = [r.score for r in self._results]
        elos = [r.elo for r in self._results]

        # Score trend: compare first 25% to last 25%
        quarter = max(1, len(scores) // 4)
        early_avg = float(np.mean(scores[:quarter]))
        late_avg = float(np.mean(scores[-quarter:]))

        return {
            "total_rounds": len(self._results),
            "final_elo": round(elos[-1], 2),
            "peak_elo": round(max(elos), 2),
            "elo_growth": round(elos[-1] - elos[0], 2),
            "mean_score": round(float(np.mean(scores)), 4),
            "score_std": round(float(np.std(scores)), 4),
            "best_score": round(max(scores), 4),
            "worst_score": round(min(scores), 4),
            "pass_rate": round(sum(1 for r in self._results if r.passed) / len(self._results), 4),
            "difficulty_advances": self._difficulty_advances,
            "score_improvement": round(late_avg - early_avg, 4),
            "early_avg_score": round(early_avg, 4),
            "late_avg_score": round(late_avg, 4),
        }

    # ══════════════════════════════════════════════════════════════════
    # Display
    # ══════════════════════════════════════════════════════════════════

    def _print_header(self) -> None:
        print()
        print("+" + "=" * 90 + "+")
        print("|  SELF-PLAY ARENA — Adaptive Curriculum Training" + " " * 42 + "|")
        print("|  Mastery gate: {} consecutive passes | Pass threshold: {:.2f}".format(
            self.mastery_window, self.pass_threshold) + " " * 23 + "|")
        print("+" + "=" * 90 + "+")
        print()
        print("  {:>5s}  {:>7s}  {:>6s}  {:>8s}  {:>7s}  {:>5s}  {:>5s}  {:>5s}  {:>6s}  {:>18s}  {:>5s}".format(
            "Round", "Score", "Pass?", "Elo", "Δ Elo", "Det", "FP", "Eff", "Streak", "Worst Weakness", "Time"))
        print("  " + "-" * 88)

    def _print_round(self, r: RoundResult) -> None:
        status = " PASS" if r.passed else " FAIL"
        delta = "{:+.1f}".format(r.elo_delta)
        print("  {:5d}  {:7.4f}  {:>6s}  {:8.1f}  {:>7s}  {:.3f}  {:.3f}  {:.3f}  {:>6d}  {:>18s}  {:4.1f}s".format(
            r.round_num, r.score, status, r.elo, delta,
            r.stats_summary["det"], r.stats_summary["fp"], r.stats_summary["eff"],
            r.mastery_streak, r.worst_weakness, r.elapsed_seconds))

    def _print_summary(self) -> None:
        summary = self._compute_summary()
        if not summary:
            return

        print()
        print("  " + "=" * 88)
        print("  TRAINING SUMMARY")
        print("  " + "=" * 88)
        print("  Rounds played:     {}".format(summary["total_rounds"]))
        print("  Final Elo:         {:.1f}  (peak: {:.1f}, growth: {:+.1f})".format(
            summary["final_elo"], summary["peak_elo"], summary["elo_growth"]))
        print("  Mean score:        {:.4f}  (std: {:.4f})".format(
            summary["mean_score"], summary["score_std"]))
        print("  Best / Worst:      {:.4f} / {:.4f}".format(
            summary["best_score"], summary["worst_score"]))
        print("  Pass rate:         {:.1%}".format(summary["pass_rate"]))
        print("  Difficulty advances: {}".format(summary["difficulty_advances"]))
        print("  Score improvement: {:+.4f}  (early {:.4f} → late {:.4f})".format(
            summary["score_improvement"], summary["early_avg_score"], summary["late_avg_score"]))
        print()
