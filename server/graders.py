"""Deterministic grading system for the three firewall tasks.

Each task has:
  - A fixed seed for reproducible traffic
  - Weighted scoring across detection, false positives, efficiency, etc.
  - A score in [0.0, 1.0] and a pass threshold

Graders are deterministic: same seed + same policy = same score.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

# Updated import path
from server.firewall_environment import FirewallEnvironment


@dataclass(frozen=True)
class TaskSpec:
    name: str
    task_key: str
    threshold: float
    weights: Dict[str, float]
    seed: int


TASK_SPECS = {
    "easy": TaskSpec(
        name="Perimeter Defense",
        task_key="easy",
        threshold=0.70,
        seed=101,
        weights={
            "detection_rate": 0.35,
            "fp_complement": 0.35,
            "efficiency": 0.30,
        },
    ),
    "medium": TaskSpec(
        name="Mixed Threat Landscape",
        task_key="medium",
        threshold=0.50,
        seed=202,
        weights={
            "detection_rate": 0.25,
            "fp_complement": 0.30,
            "efficiency": 0.15,
            "early_detection_bonus": 0.15,
            "cascade_prevention": 0.15,
        },
    ),
    "hard": TaskSpec(
        name="Advanced Persistent Threat",
        task_key="hard",
        threshold=0.45,
        seed=303,
        weights={
            "detection_rate": 0.20,
            "fp_complement": 0.25,
            "efficiency": 0.15,
            "early_detection_bonus": 0.20,
            "cascade_prevention": 0.20,
        },
    ),
}

PASS_CONSTRAINTS = {
    "easy": {"min_detection_rate": 0.35, "min_fp_complement": 0.65},
    "medium": {"min_detection_rate": 0.35, "min_fp_complement": 0.60},
    "hard": {"min_detection_rate": 0.35, "min_fp_complement": 0.55},
}


def grade_stats(task: str, stats: Dict) -> Dict:
    """Compute a grade from episode stats."""
    spec = TASK_SPECS[task]
    values = {
        "detection_rate": stats.get("detection_rate", 0.0),
        "fp_complement": 1.0 - stats.get("false_positive_rate", 1.0),
        "efficiency": stats.get("efficiency", 0.0),
        "early_detection_bonus": stats.get("early_detection_bonus", 0.0),
        "cascade_prevention": stats.get("cascade_prevention", 0.0),
    }
    score = sum(values.get(k, 0.0) * w for k, w in spec.weights.items())
    score = max(0.0, min(1.0, score))
    constraints = PASS_CONSTRAINTS[task]
    meets_constraints = (
        values["detection_rate"] >= constraints["min_detection_rate"]
        and values["fp_complement"] >= constraints["min_fp_complement"]
    )
    passed = (score >= spec.threshold) and meets_constraints

    return {
        "task": task,
        "task_name": spec.name,
        "threshold": spec.threshold,
        "score": round(score, 6),
        "passed": passed,
        "pass_constraints": constraints,
        "meets_constraints": meets_constraints,
        "breakdown": {k: round(v, 6) for k, v in values.items()},
    }


def run_deterministic_grade(
    env: FirewallEnvironment,
    task: str,
    policy: Callable[[FirewallEnvironment, List[str]], Dict[str, int]],
) -> Dict:
    """Run a full episode with a policy and compute the grade."""
    spec = TASK_SPECS[task]
    env.reset(task=task, seed=spec.seed)
    done = False
    while not done:
        session_ids = (
            list(env.inspected_sessions.keys())
            + list(env.pending_sessions.keys())
        )
        actions = policy(env, session_ids)
        response = env.step(actions)
        done = bool(response["done"])
    stats = env.get_network_stats()
    return grade_stats(task, stats)
