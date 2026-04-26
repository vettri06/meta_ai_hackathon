"""Multi-stage attack orchestrator following Cyber Kill Chain model.

Each attacker has a scenario (one of 5 patterns) and progresses through
phases 0→3.  Adaptation is non-trivial:
  - Detected attackers may switch to stealth mode (mimic benign profiles)
  - Undetected attackers escalate normally
  - Fully blocked attackers are terminated
  - Attackers that reach exfiltration (phase 3) are marked as succeeded
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import numpy as np

# Updated import path
from server.utils.data_loader import TrafficGenerator


SCENARIOS = [
    "port_scan_exploit_c2",
    "credential_stuffing_lateral",
    "supply_chain_compromise",
    "low_and_slow_apt",
    "ddos_amplification",
]

# How many sessions each scenario generates per phase
SESSION_COUNTS: Dict[str, List[int]] = {
    "port_scan_exploit_c2":        [4, 2, 1, 2],
    "credential_stuffing_lateral": [3, 3, 2, 2],
    "supply_chain_compromise":     [1, 1, 1, 2],
    "low_and_slow_apt":            [1, 1, 1, 1],
    "ddos_amplification":          [6, 10, 15, 20],
}

# Probability that an attacker escalates per tick (if not detected)
ESCALATION_PROB: Dict[str, float] = {
    "port_scan_exploit_c2":        0.30,
    "credential_stuffing_lateral": 0.25,
    "supply_chain_compromise":     0.15,
    "low_and_slow_apt":            0.10,
    "ddos_amplification":          0.40,
}


@dataclass
class AttackerState:
    attacker_id: str
    scenario: str
    phase: int = 0
    times_detected: int = 0
    stealth_mode: bool = False
    alive: bool = True
    succeeded: bool = False
    ticks_alive: int = 0
    sessions_blocked: int = 0
    sessions_generated: int = 0


class ThreatEngine:
    """Manages the lifecycle of active attackers and generates attack sessions."""

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self._attacker_counter = 0
        self._active_attackers: Dict[str, AttackerState] = {}
        self._dead_attackers: List[AttackerState] = []
        self._threat_intel: Dict = {
            "known_bad_ports": [21, 22, 23, 25, 445, 3389, 5900],
            "known_bad_ja3_ranges": [(200, 255), (230, 255)],
            "active_campaigns": [],
            "recent_detections": 0,
        }

    def reset(self) -> None:
        self._attacker_counter = 0
        self._active_attackers = {}
        self._dead_attackers = []
        self._threat_intel["active_campaigns"] = []
        self._threat_intel["recent_detections"] = 0

    def maybe_spawn_attacker(self, threat_probability: float) -> None:
        """Probabilistically spawn a new attacker."""
        if self.rng.random() > threat_probability:
            return
        self._attacker_counter += 1
        scenario = SCENARIOS[int(self.rng.integers(0, len(SCENARIOS)))]
        attacker_id = f"a-{self._attacker_counter:04d}"
        state = AttackerState(attacker_id=attacker_id, scenario=scenario)
        self._active_attackers[attacker_id] = state
        # Update threat intel
        campaigns = set(self._threat_intel["active_campaigns"])
        campaigns.add(scenario)
        self._threat_intel["active_campaigns"] = sorted(campaigns)

    def generate_attack_sessions(
        self, tick: int, generator: TrafficGenerator,
        blocked_attackers: Set[str],
        escalation_rate_mod: float = 1.0,
        stealth_multiplier: float = 1.0,
    ) -> List[Dict]:
        """Generate attack sessions for all active attackers, handling adaptation.

        Args:
            escalation_rate_mod: Multiplier on base escalation probability.
                Values > 1.0 make attackers advance through kill chain faster.
            stealth_multiplier: Controls how hard stealth attacks are to detect.
                Values > 1.0 blend malicious features toward benign distributions.
        """
        sessions: List[Dict] = []

        for attacker in list(self._active_attackers.values()):
            if not attacker.alive:
                continue

            attacker.ticks_alive += 1

            # --- Handle detection / blocking ---
            if attacker.attacker_id in blocked_attackers:
                attacker.times_detected += 1
                attacker.sessions_blocked += 1
                self._threat_intel["recent_detections"] += 1

                if attacker.times_detected >= 3:
                    # Fully blocked — attacker gives up
                    attacker.alive = False
                    self._dead_attackers.append(attacker)
                    continue
                elif attacker.times_detected >= 2:
                    # Switch to stealth mode — generate fewer, more benign-looking sessions
                    attacker.stealth_mode = True
                else:
                    # First detection — try to advance past detected phase
                    attacker.phase = min(attacker.phase + 1, 3)

            # --- Natural phase escalation (modified by escalation_rate_mod) ---
            else:
                base_prob = ESCALATION_PROB.get(attacker.scenario, 0.2)
                effective_prob = min(0.95, base_prob * escalation_rate_mod)
                if self.rng.random() < effective_prob:
                    attacker.phase = min(attacker.phase + 1, 3)

            # --- Check for success (exfiltration complete) ---
            if attacker.phase == 3 and attacker.ticks_alive > 8:
                if self.rng.random() < 0.15:
                    attacker.succeeded = True
                    attacker.alive = False
                    self._dead_attackers.append(attacker)
                    continue

            # --- Generate sessions based on current state ---
            counts = SESSION_COUNTS.get(attacker.scenario, [2, 2, 2, 2])
            count = counts[min(attacker.phase, 3)]

            if attacker.stealth_mode:
                # In stealth mode: reduce count, use profiles that look more benign
                count = max(1, count // 2)

            generated = generator.generate_malicious_sessions(
                tick=tick,
                count=count,
                attack_phase=attacker.phase,
                scenario=attacker.scenario,
                attacker_id=attacker.attacker_id,
            )

            # --- Apply stealth blending to stealth-mode sessions ---
            if attacker.stealth_mode and stealth_multiplier > 1.0:
                blend_strength = min(0.6, 0.2 * (stealth_multiplier - 1.0))
                for s in generated:
                    s["metadata"]["is_stealth"] = True
                    feats = s["features"]
                    # Blend suspicious features toward benign ranges
                    feats["session_history_score"] = float(min(1.0,
                        feats["session_history_score"] + blend_strength * 0.5
                    ))
                    feats["connection_reuse"] = float(min(1.0,
                        feats["connection_reuse"] + blend_strength * 0.4
                    ))
                    feats["entropy_score"] = float(max(0.0,
                        feats["entropy_score"] - blend_strength * 0.3
                    ))
                    feats["geo_distance"] = float(max(0.0,
                        feats["geo_distance"] * (1.0 - blend_strength * 0.4)
                    ))
            else:
                for s in generated:
                    s["metadata"]["is_stealth"] = False

            attacker.sessions_generated += len(generated)
            sessions.extend(generated)

        return sessions

    def intelligence_feed(self) -> Dict:
        """Return threat intelligence available to the agent."""
        active_scenarios = set()
        for a in self._active_attackers.values():
            if a.alive:
                active_scenarios.add(a.scenario)
        self._threat_intel["active_campaigns"] = sorted(active_scenarios)
        return dict(self._threat_intel)

    def attacker_outcomes(self) -> Dict[str, str]:
        """Return status of all known attackers (for info/debugging)."""
        outcomes: Dict[str, str] = {}
        for a in self._active_attackers.values():
            if a.alive:
                outcomes[a.attacker_id] = "active"
            elif a.succeeded:
                outcomes[a.attacker_id] = "succeeded"
            else:
                outcomes[a.attacker_id] = "stopped"
        for a in self._dead_attackers:
            if a.attacker_id not in outcomes:
                outcomes[a.attacker_id] = "succeeded" if a.succeeded else "stopped"
        return outcomes