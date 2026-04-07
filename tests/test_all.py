"""Comprehensive tests for the Adaptive AI Firewall environment.

Covers: feature generation, reward mechanics, threat lifecycle,
grading determinism, degenerate policy detection, and budget management.
"""
import numpy as np

from server.utils.data_loader import FEATURE_ORDER, TrafficGenerator
from server.utils.threat_engine import ThreatEngine
from server.utils.reward_engine import RewardEngine
from server.firewall_environment import (
    FirewallEnvironment, OBS_DIM, NUM_ACTIONS,
)
from server.graders import run_deterministic_grade, grade_stats
from server.baseline.random_agent import random_policy, block_all_policy
from server.baseline.heuristic_agent import heuristic_policy


# ═══════════════════════════════════════════════════════════════════
# Traffic Generator
# ═══════════════════════════════════════════════════════════════════

class TestTrafficGenerator:
    def test_feature_dimension(self):
        gen = TrafficGenerator(seed=11)
        session = gen.generate_benign_sessions(tick=0, count=1)[0]
        assert len(FEATURE_ORDER) == 22
        assert len(gen.to_observation_vector(session)) == 22

    def test_normalized_features_in_0_1(self):
        gen = TrafficGenerator(seed=42)
        for _ in range(50):
            session = gen.generate_benign_sessions(tick=0, count=1)[0]
            obs = gen.to_observation_vector(session)
            for i, val in enumerate(obs):
                assert 0.0 <= val <= 1.0, f"Feature {FEATURE_ORDER[i]} = {val} out of [0,1]"

    def test_malicious_features_normalized(self):
        gen = TrafficGenerator(seed=55)
        for scenario in ["port_scan_exploit_c2", "ddos_amplification", "supply_chain_compromise"]:
            for phase in range(4):
                sessions = gen.generate_malicious_sessions(
                    tick=0, count=3, attack_phase=phase, scenario=scenario,
                )
                for s in sessions:
                    obs = gen.to_observation_vector(s)
                    for i, val in enumerate(obs):
                        assert 0.0 <= val <= 1.0

    def test_benign_malicious_separation(self):
        """Verify that malicious and benign sessions have statistically different features."""
        gen = TrafficGenerator(seed=77)
        benign_vecs = []
        for _ in range(100):
            s = gen.generate_benign_sessions(tick=0, count=1)[0]
            benign_vecs.append(gen.to_observation_vector(s))

        mal_vecs = []
        for phase in range(4):
            for _ in range(25):
                s = gen.generate_malicious_sessions(
                    tick=0, count=1, attack_phase=phase,
                    scenario="port_scan_exploit_c2",
                )[0]
                mal_vecs.append(gen.to_observation_vector(s))

        benign_arr = np.array(benign_vecs)
        mal_arr = np.array(mal_vecs)

        # At least some features should have meaningfully different means
        mean_diff = np.abs(benign_arr.mean(axis=0) - mal_arr.mean(axis=0))
        significant_features = (mean_diff > 0.08).sum()
        assert significant_features >= 5, (
            f"Only {significant_features} features differ — distributions too similar"
        )

    def test_session_ids_unique(self):
        gen = TrafficGenerator(seed=99)
        ids = set()
        for _ in range(100):
            sessions = gen.generate_benign_sessions(tick=0, count=3)
            for s in sessions:
                assert s["session_id"] not in ids
                ids.add(s["session_id"])


# ═══════════════════════════════════════════════════════════════════
# Reward Engine
# ═══════════════════════════════════════════════════════════════════

class TestRewardEngine:
    def test_block_malicious_positive(self):
        eng = RewardEngine()
        r, _ = eng.reward(action=1, is_malicious=True, budget_remaining=50.0, attack_phase=0)
        assert r > 0

    def test_miss_malicious_negative(self):
        eng = RewardEngine()
        r, _ = eng.reward(action=0, is_malicious=True, budget_remaining=50.0, attack_phase=2)
        assert r < 0

    def test_block_benign_negative(self):
        eng = RewardEngine()
        r, _ = eng.reward(action=1, is_malicious=False, budget_remaining=50.0, attack_phase=0)
        assert r < 0

    def test_allow_benign_positive(self):
        eng = RewardEngine()
        r, _ = eng.reward(action=0, is_malicious=False, budget_remaining=50.0, attack_phase=0)
        assert r > 0, "Correctly allowing benign traffic should be rewarded"

    def test_block_all_loses_in_mixed_traffic(self):
        """Block-all should have negative total reward on benign-heavy traffic."""
        eng = RewardEngine()
        total = 0.0
        # Simulate 80% benign, 20% malicious
        for _ in range(80):
            r, _ = eng.reward(action=1, is_malicious=False, budget_remaining=50.0, attack_phase=0)
            total += r
        for _ in range(20):
            r, _ = eng.reward(action=1, is_malicious=True, budget_remaining=50.0, attack_phase=1)
            total += r
        # Block-all should have lower score than a selective policy
        assert total < 0, f"Block-all total reward {total} should be negative on 80/20 mix"

    def test_early_detection_bonus(self):
        eng = RewardEngine()
        r_early, _ = eng.reward(action=1, is_malicious=True, budget_remaining=50.0, attack_phase=0)
        r_late, _ = eng.reward(action=1, is_malicious=True, budget_remaining=50.0, attack_phase=3)
        assert r_early > r_late, "Early detection should give higher reward"


# ═══════════════════════════════════════════════════════════════════
# Threat Engine
# ═══════════════════════════════════════════════════════════════════

class TestThreatEngine:
    def test_spawn_and_generate(self):
        engine = ThreatEngine(seed=22)
        gen = TrafficGenerator(seed=23)
        engine.maybe_spawn_attacker(1.0)
        sessions = engine.generate_attack_sessions(tick=0, generator=gen, blocked_attackers=set())
        assert len(sessions) > 0
        assert all(s["metadata"]["malicious"] for s in sessions)

    def test_attacker_dies_after_3_blocks(self):
        engine = ThreatEngine(seed=33)
        gen = TrafficGenerator(seed=34)
        engine.maybe_spawn_attacker(1.0)
        attacker_id = list(engine._active_attackers.keys())[0]

        for _ in range(3):
            engine.generate_attack_sessions(
                tick=0, generator=gen, blocked_attackers={attacker_id},
            )

        # After 3 blocks, attacker should be dead
        attacker = engine._active_attackers[attacker_id]
        assert not attacker.alive

    def test_attacker_outcomes(self):
        engine = ThreatEngine(seed=44)
        gen = TrafficGenerator(seed=45)
        engine.maybe_spawn_attacker(1.0)
        engine.generate_attack_sessions(tick=0, generator=gen, blocked_attackers=set())
        outcomes = engine.attacker_outcomes()
        assert len(outcomes) > 0
        assert all(v in ("active", "stopped", "succeeded") for v in outcomes.values())


# ═══════════════════════════════════════════════════════════════════
# Firewall Environment
# ═══════════════════════════════════════════════════════════════════

class TestFirewallEnvironment:
    def test_reset_returns_valid_state(self):
        env = FirewallEnvironment(seed=99)
        state = env.reset(task="easy", seed=100)
        assert state["observation_dim"] == OBS_DIM
        assert state["num_actions"] == NUM_ACTIONS
        assert state["budget_remaining"] > 0

    def test_step_returns_expected_keys(self):
        env = FirewallEnvironment(seed=99)
        env.reset(task="easy", seed=100)
        pending = list(env.pending_sessions.keys())
        actions = {sid: 0 for sid in pending[:3]}
        response = env.step(actions)
        assert "reward" in response
        assert "done" in response
        assert "state" in response

    def test_inspect_keeps_session_alive(self):
        env = FirewallEnvironment(seed=50)
        env.reset(task="easy", seed=50)
        sid = list(env.pending_sessions.keys())[0]
        env._apply_action(sid, 2)  # INSPECT
        assert sid in env.inspected_sessions, "INSPECT should keep session in inspected pool"

    def test_inspect_then_block(self):
        """Two-phase: inspect → block."""
        env = FirewallEnvironment(seed=60)
        env.reset(task="easy", seed=60)
        sid = list(env.pending_sessions.keys())[0]

        # Phase 1: inspect
        r1, _ = env._apply_action(sid, 2)
        assert sid in env.inspected_sessions

        # Phase 2: block
        r2, _ = env._apply_action(sid, 1)
        assert sid not in env.inspected_sessions

    def test_budget_stays_positive_with_allow(self):
        """All-allow policy should preserve most of the budget."""
        env = FirewallEnvironment(seed=70)
        env.reset(task="easy", seed=70)
        initial = env.budget_remaining
        for _ in range(50):
            sids = list(env.pending_sessions.keys())
            if not sids:
                break
            env.step({sid: 0 for sid in sids})
        # ALLOW costs 0, so budget should barely change
        assert env.budget_remaining >= initial * 0.95

    def test_budget_nonzero_with_reasonable_policy(self):
        """Heuristic policy should leave some budget remaining."""
        env = FirewallEnvironment(seed=80)
        env.reset(task="easy", seed=80)
        for _ in range(env.max_steps):
            sids = (
                list(env.inspected_sessions.keys())
                + list(env.pending_sessions.keys())
            )
            actions = heuristic_policy(env, sids)
            resp = env.step(actions)
            if resp["done"]:
                break
        stats = env.get_network_stats()
        assert stats["efficiency"] > 0.0, f"Efficiency should be > 0, got {stats['efficiency']}"

    def test_expired_malicious_counted_in_metrics(self):
        """Expired malicious sessions must be counted in totals."""
        env = FirewallEnvironment(seed=90)
        env.reset(task="easy", seed=90)
        # Let everything expire by stepping with no actions
        for _ in range(10):
            env.step({})
        stats = env.get_network_stats()
        if stats["total_malicious"] > 0:
            # expired malicious should be counted
            assert stats["expired_malicious"] > 0

    def test_single_session_mode(self):
        """step_single returns valid observation and reward."""
        env = FirewallEnvironment(seed=100)
        env.reset(task="easy", seed=100)
        result = env.step_single(0)  # ALLOW
        assert len(result["observation"]) == OBS_DIM
        assert "reward" in result
        assert "done" in result


# ═══════════════════════════════════════════════════════════════════
# Graders
# ═══════════════════════════════════════════════════════════════════

class TestGraders:
    def test_deterministic_grading(self):
        env = FirewallEnvironment(seed=31)
        p1 = random_policy(seed=9)
        first = run_deterministic_grade(env, task="easy", policy=p1)["score"]
        p2 = random_policy(seed=9)
        second = run_deterministic_grade(env, task="easy", policy=p2)["score"]
        assert first == second, "Same seed should produce same score"

    def test_score_in_valid_range(self):
        env = FirewallEnvironment(seed=40)
        for task in ("easy", "medium", "hard"):
            policy = random_policy(seed=7)
            result = run_deterministic_grade(env, task=task, policy=policy)
            assert 0.0 <= result["score"] <= 1.0

    def test_heuristic_beats_random(self):
        """Core sanity check: heuristic > random on easy task."""
        env = FirewallEnvironment(seed=50)
        rp = random_policy(seed=7)
        r_score = run_deterministic_grade(env, task="easy", policy=rp)["score"]
        h_score = run_deterministic_grade(env, task="easy", policy=heuristic_policy)["score"]
        assert h_score > r_score, (
            f"Heuristic ({h_score:.4f}) must beat random ({r_score:.4f}) on easy task"
        )

    def test_heuristic_beats_block_all(self):
        """Block-all should not dominate heuristic."""
        env = FirewallEnvironment(seed=60)
        b_score = run_deterministic_grade(env, task="easy", policy=block_all_policy)["score"]
        h_score = run_deterministic_grade(env, task="easy", policy=heuristic_policy)["score"]
        assert h_score > b_score, (
            f"Heuristic ({h_score:.4f}) must beat block-all ({b_score:.4f})"
        )

    def test_grade_stats_clamps(self):
        stats = {"detection_rate": 1.5, "false_positive_rate": -0.5, "efficiency": 2.0}
        result = grade_stats("easy", stats)
        assert result["score"] <= 1.0
