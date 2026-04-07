from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Updated imports to reflect new structure
from server.utils.reward_engine import (
    ACTIONS, BLOCKING_ACTIONS, RewardEngine,
)
from server.utils.threat_engine import ThreatEngine
from server.utils.data_loader import (
    FEATURE_ORDER, TrafficGenerator,
)


TASK_CONFIGS = {
    "easy": {
        "max_steps": 200,
        "benign_ratio": 0.80,
        "threat_probability": 0.12,
        "traffic_lambda": 5,
        "budget": 100.0,        # ~0.50 budget per step
    },
    "medium": {
        "max_steps": 500,
        "benign_ratio": 0.65,
        "threat_probability": 0.22,
        "traffic_lambda": 6,
        "budget": 300.0,       # ~0.60 budget per step
    },
    "hard": {
        "max_steps": 1000,
        "benign_ratio": 0.70,
        "threat_probability": 0.30,
        "traffic_lambda": 7,
        "budget": 600.0,       # ~0.60 budget per step
    },
}

NUM_ACTIONS = len(ACTIONS)
OBS_DIM = len(FEATURE_ORDER)


@dataclass
class EpisodeMetrics:
    """Tracks all metrics needed for grading."""
    detections: int = 0
    malicious_seen: int = 0
    false_positives: int = 0
    benign_seen: int = 0
    early_detection_sum: float = 0.0
    cascade_failures: int = 0
    total_cost: float = 0.0
    sessions_expired_malicious: int = 0
    sessions_expired_benign: int = 0
    correct_allows: int = 0
    inspections: int = 0


class FirewallEnvironment:
    """Adaptive AI Firewall RL environment.

    OpenEnv-compatible: reset(), step(), state()

    Key design (from RL perspective):
      - Observation: 22-dim normalized [0,1] vector per session
      - Action: Discrete(6) — ALLOW, BLOCK, INSPECT, SANDBOX, RATE_LIMIT, QUARANTINE
      - Reward: multi-objective (security + availability + efficiency + timeliness)
      - Done: when max_steps reached or budget depleted
      - INSPECT keeps session alive for a second action (two-phase decision)
    """

    def __init__(self, seed: int = 0, budget: Optional[float] = None) -> None:
        self.base_seed = seed
        self.base_budget_override = budget
        self.generator = TrafficGenerator(seed=seed)
        self.threat_engine = ThreatEngine(seed=seed + 1)
        self.reward_engine = RewardEngine()
        self.rng = np.random.default_rng(seed + 2)

        self.episode_id = 0
        self.step_count = 0
        self.current_tick = 0
        self.task = "easy"
        self.max_steps = TASK_CONFIGS[self.task]["max_steps"]
        default_budget = TASK_CONFIGS[self.task]["budget"]
        if self.base_budget_override is not None:
            default_budget = max(default_budget, float(self.base_budget_override))
        self.budget_remaining = default_budget
        self.initial_budget = self.budget_remaining
        self.total_reward = 0.0

        self.pending_sessions: Dict[str, Dict] = {}
        self.inspected_sessions: Dict[str, Dict] = {}  # sessions awaiting 2nd action
        self.action_log: List[Dict] = []
        self._blocked_attacker_ids: Set[str] = set()
        self.metrics = EpisodeMetrics()

        # For single-session mode
        self._session_queue: List[str] = []

    # ══════════════════════════════════════════════════════════════════
    # OpenEnv API
    # ══════════════════════════════════════════════════════════════════

    def reset(self, task: str = "easy", seed: Optional[int] = None) -> Dict:
        """Reset environment for a new episode."""
        if task not in TASK_CONFIGS:
            raise ValueError(f"unknown task: {task}")

        used_seed = self.base_seed if seed is None else seed
        self.generator = TrafficGenerator(seed=used_seed)
        self.threat_engine = ThreatEngine(seed=used_seed + 1)
        self.rng = np.random.default_rng(used_seed + 2)

        self.episode_id += 1
        self.step_count = 0
        self.current_tick = 0
        self.task = task
        config = TASK_CONFIGS[task]
        self.max_steps = config["max_steps"]
        task_budget = float(config["budget"])
        if self.base_budget_override is not None:
            task_budget = max(task_budget, float(self.base_budget_override))
        self.initial_budget = task_budget
        self.budget_remaining = self.initial_budget
        self.total_reward = 0.0

        self.pending_sessions = {}
        self.inspected_sessions = {}
        self.action_log = []
        self._blocked_attacker_ids = set()
        self.metrics = EpisodeMetrics()
        self._session_queue = []

        # Spawn initial sessions
        self._spawn_sessions()
        self._rebuild_queue()

        return self.state()

    def step(self, action_map: Optional[Dict[str, int]] = None) -> Dict:
        """Multi-session step: agent provides actions for multiple sessions at once."""
        action_map = action_map or {}
        step_reward = 0.0

        for session_id, action in action_map.items():
            # Check both pending and inspected pools
            if session_id in self.pending_sessions or session_id in self.inspected_sessions:
                reward, _ = self._apply_action(session_id, action)
                step_reward += reward

        expired_penalty = self._expire_sessions()
        step_reward += expired_penalty
        self.total_reward += step_reward
        self.step_count += 1
        self.current_tick += 1

        done = self.step_count >= self.max_steps or self.budget_remaining <= 0.0

        if not done:
            self._spawn_sessions()
            self._rebuild_queue()

        # Calculate score using the deterministic grader logic
        final_stats = self.get_network_stats()
        from server.graders import grade_stats
        grade = grade_stats(self.task, final_stats)
        return {
            "reward": step_reward,
            "done": done,
            "state": self.state(),
            "info": {
                "expired_penalty": expired_penalty,
                "attacker_outcomes": self.threat_engine.attacker_outcomes(),
                "score": grade["score"],
                "passed": grade["passed"]
            },
        }

    def step_single(self, action: int) -> Dict:
        """Single-session step: present one session, agent picks one action.

        Compatible with Gymnasium Discrete(6).
        Returns observation of the NEXT session, or zeros if episode done.
        """
        if action not in ACTIONS:
            raise ValueError(f"invalid action: {action}")

        step_reward = 0.0
        info: Dict[str, Any] = {}

        # Act on the current session
        if self._session_queue:
            session_id = self._session_queue.pop(0)
            if session_id in self.pending_sessions or session_id in self.inspected_sessions:
                reward, record = self._apply_action(session_id, action)
                step_reward += reward
                info["action_record"] = record

        self.total_reward = round(self.total_reward + step_reward, 4)
        self.step_count += 1

        # If queue is empty, advance tick
        if not self._session_queue:
            self.current_tick += 1
            expired_penalty = self._expire_sessions()
            # step_reward for the final session in tick includes the expiration penalty
            step_reward += expired_penalty
            self.total_reward = round(self.total_reward + expired_penalty, 4)
            done = self.step_count >= self.max_steps or self.budget_remaining <= 0.0
            if not done:
                self._spawn_sessions()
                self._rebuild_queue()
        else:
            done = self.step_count >= self.max_steps or self.budget_remaining <= 0.0

        # Build next observation
        next_obs = self._current_observation()

        return {
            "observation": next_obs,
            "reward": step_reward,
            "done": done,
            "state": {
                **self.state(),
                "focus_observation": next_obs,
                "focus_session_id": self._session_queue[0] if self._session_queue else None,
            },
            "info": info,
        }

    def state(self) -> Dict:
        """Return current environment state (OpenEnv API)."""
        all_sessions = {**self.pending_sessions, **self.inspected_sessions}
        top_ids = list(all_sessions.keys())[:10]
        focus_session_id = self._session_queue[0] if self._session_queue else None
        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "step_count": self.step_count,
            "current_tick": self.current_tick,
            "observation_dim": OBS_DIM,
            "num_actions": NUM_ACTIONS,
            "budget_remaining": round(self.budget_remaining, 4),
            "total_reward": round(self.total_reward, 4),
            "pending_session_count": len(self.pending_sessions),
            "inspected_session_count": len(self.inspected_sessions),
            "pending_session_ids": top_ids,
            "inspected_session_ids": list(self.inspected_sessions.keys())[:10],
            "queue_length": len(self._session_queue),
            "focus_session_id": focus_session_id,
            "focus_observation": self._current_observation(),
        }

    # ══════════════════════════════════════════════════════════════════
    # Tool API (for MCP/HTTP interface)
    # ══════════════════════════════════════════════════════════════════

    def evaluate_session(self, session_id: str) -> Dict:
        """Get observation vector and metadata for a session."""
        session = self.pending_sessions.get(session_id) or self.inspected_sessions.get(session_id)
        if session is None:
            raise KeyError(f"session not found: {session_id}")

        return {
            "session_id": session_id,
            "features": dict(session["features"]),
            "observation": self.generator.to_observation_vector(session),
            "is_inspected": session_id in self.inspected_sessions,
            "revealed_malicious": (
                session["metadata"]["malicious"]
                if session["metadata"]["revealed"] else None
            ),
            "expires_tick": session["expires_tick"],
        }

    def take_action(self, session_id: str, action: int) -> Tuple[float, Dict]:
        """Apply an action to a specific session."""
        return self._apply_action(session_id, action)

    def get_network_stats(self) -> Dict:
        """Aggregate episode statistics for grading."""
        m = self.metrics
        total_malicious = m.malicious_seen + m.sessions_expired_malicious
        total_benign = m.benign_seen + m.sessions_expired_benign

        detection_rate = m.detections / max(total_malicious, 1)
        false_positive_rate = m.false_positives / max(total_benign, 1)
        efficiency = 1.0 - min(1.0, m.total_cost / max(self.initial_budget, 1e-6))
        early_detection_bonus = m.early_detection_sum / max(m.detections, 1)
        cascade_prevention = 1.0 - (m.cascade_failures / max(total_malicious, 1))

        return {
            "episode_id": self.episode_id,
            "task": self.task,
            "tick": self.current_tick,
            "step_count": self.step_count,
            "total_reward": round(self.total_reward, 4),
            "budget_remaining": round(self.budget_remaining, 4),
            "budget_used_pct": round(1.0 - self.budget_remaining / max(self.initial_budget, 1e-6), 4),
            "total_malicious": total_malicious,
            "total_benign": total_benign,
            "detection_rate": round(detection_rate, 6),
            "false_positive_rate": round(false_positive_rate, 6),
            "efficiency": round(efficiency, 6),
            "early_detection_bonus": round(early_detection_bonus, 6),
            "cascade_prevention": round(cascade_prevention, 6),
            "correct_allows": m.correct_allows,
            "inspections": m.inspections,
            "expired_malicious": m.sessions_expired_malicious,
            "expired_benign": m.sessions_expired_benign,
        }

    def get_threat_intelligence(self) -> Dict:
        return self.threat_engine.intelligence_feed()

    def list_tools(self) -> List[str]:
        return [
            "evaluate_session", "take_action",
            "get_network_stats", "get_threat_intelligence",
        ]

    # ══════════════════════════════════════════════════════════════════
    # Internal mechanics
    # ══════════════════════════════════════════════════════════════════

    def _apply_action(self, session_id: str, action: int) -> Tuple[float, Dict]:
        """Core action application logic."""
        if action not in ACTIONS:
            raise ValueError(f"invalid action: {action}")

        # Find the session in either pool
        source_pool = "none"
        if session_id in self.inspected_sessions:
            session = self.inspected_sessions.pop(session_id)
            source_pool = "inspected"
        elif session_id in self.pending_sessions:
            session = self.pending_sessions.pop(session_id)
            source_pool = "pending"
        else:
            raise KeyError(f"session not found: {session_id}")

        metadata = session["metadata"]
        malicious = bool(metadata["malicious"])
        blocked = action in BLOCKING_ACTIONS
        inspected = action == 2  # INSPECT

        # ── INSPECT keeps the session alive for a second decision ──
        if inspected and session_id not in self.inspected_sessions:
            metadata["revealed"] = True
            self.inspected_sessions[session_id] = session
            self.pending_sessions[session_id] = session
            self.metrics.inspections += 1
            # Compute reward for the inspection itself
            reward, components = self.reward_engine.reward(
                action=action,
                is_malicious=malicious,
                budget_remaining=self.budget_remaining,
                attack_phase=metadata.get("attack_phase", 0),
                inspect_correct=malicious,
            )
            self.budget_remaining = max(0.0, self.budget_remaining - components["cost"])
            self.metrics.total_cost += components["cost"]
            record = self._make_record(session_id, action, malicious, reward, components)
            return reward, record

        # ── Terminal action (ALLOW, BLOCK, SANDBOX, RATE_LIMIT, QUARANTINE) ──
        inspect_correct = malicious and metadata.get("revealed", False)
        reward, components = self.reward_engine.reward(
            action=action,
            is_malicious=malicious,
            budget_remaining=self.budget_remaining,
            attack_phase=metadata.get("attack_phase", 0),
            inspect_correct=inspect_correct,
        )
        self.budget_remaining = max(0.0, self.budget_remaining - components["cost"])
        self.metrics.total_cost += components["cost"]
        if source_pool == "inspected":
            self.pending_sessions.pop(session_id, None)

        # ── Update metrics ──
        if malicious:
            self.metrics.malicious_seen += 1
            if blocked:
                self.metrics.detections += 1
                phase = metadata.get("attack_phase", 0)
                self.metrics.early_detection_sum += float(np.exp(-phase))
                attacker_id = metadata.get("attacker_id")
                if attacker_id:
                    self._blocked_attacker_ids.add(attacker_id)
            else:
                if metadata.get("attack_phase", 0) >= 2:
                    self.metrics.cascade_failures += 1
        else:
            self.metrics.benign_seen += 1
            if blocked:
                self.metrics.false_positives += 1
            elif action == 0:
                self.metrics.correct_allows += 1

        record = self._make_record(session_id, action, malicious, reward, components)
        self.action_log.append(record)
        return reward, record

    def _make_record(self, session_id: str, action: int, malicious: bool,
                     reward: float, components: Dict) -> Dict:
        return {
            "tick": self.current_tick,
            "session_id": session_id,
            "action": action,
            "action_name": ACTIONS[action],
            "malicious": malicious,
            "reward": round(reward, 6),
            "components": {k: round(v, 6) for k, v in components.items()},
        }

    def _spawn_sessions(self) -> None:
        """Generate new benign and malicious sessions for current tick."""
        config = TASK_CONFIGS[self.task]
        benign_count = int(max(1, self.rng.poisson(
            config["traffic_lambda"] * config["benign_ratio"],
        )))
        benign = self.generator.generate_benign_sessions(
            tick=self.current_tick, count=benign_count,
        )

        self.threat_engine.maybe_spawn_attacker(config["threat_probability"])
        malicious = self.threat_engine.generate_attack_sessions(
            tick=self.current_tick,
            generator=self.generator,
            blocked_attackers=self._blocked_attacker_ids,
        )
        self._blocked_attacker_ids = set()

        for session in benign + malicious:
            self.pending_sessions[session["session_id"]] = session

    def _expire_sessions(self) -> float:
        """Remove expired sessions and apply penalties. Count in metrics."""
        expired_ids = set()
        for sid, session in self.pending_sessions.items():
            if session["expires_tick"] <= self.current_tick:
                expired_ids.add(sid)
        for sid, session in self.inspected_sessions.items():
            if session["expires_tick"] <= self.current_tick:
                expired_ids.add(sid)
        penalty = 0.0
        for session_id in expired_ids:
            session = self.inspected_sessions.pop(session_id, None)
            if session is None:
                session = self.pending_sessions.get(session_id)
            self.pending_sessions.pop(session_id, None)
            if session is None:
                continue
            if session["metadata"]["malicious"]:
                penalty -= 1.5
                self.metrics.sessions_expired_malicious += 1
                if session["metadata"].get("attack_phase", 0) >= 2:
                    self.metrics.cascade_failures += 1
            else:
                self.metrics.sessions_expired_benign += 1

        return penalty

    def _rebuild_queue(self) -> None:
        """Rebuild the single-session queue from pending + inspected."""
        # Inspected sessions get priority (they need a follow-up action)
        ordered = list(self.inspected_sessions.keys()) + list(self.pending_sessions.keys())
        seen: Set[str] = set()
        self._session_queue = []
        for sid in ordered:
            if sid in seen:
                continue
            seen.add(sid)
            self._session_queue.append(sid)

    def _current_observation(self) -> List[float]:
        """Get normalized observation for the next session in queue."""
        if self._session_queue:
            sid = self._session_queue[0]
            session = (
                self.inspected_sessions.get(sid)
                or self.pending_sessions.get(sid)
            )
            if session:
                return self.generator.to_observation_vector(session)
        return [0.0] * OBS_DIM
