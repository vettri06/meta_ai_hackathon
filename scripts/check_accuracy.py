"""Comprehensive accuracy and parameter validation for the Firewall environment.

Runs all tasks x all policies and validates:
  1. New parameter effects (noise, stealth, bursts, false flags, TTL, escalation)
  2. Grading accuracy across difficulty tiers
  3. Policy ordering: heuristic > block-all > random
  4. Feature distribution separation (benign vs malicious)
  5. Environment invariants (budget, observations, session lifecycle)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Force UTF-8 output on Windows
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
if sys.stdout.encoding != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from server.firewall_environment import FirewallEnvironment, TASK_CONFIGS
from server.graders import run_deterministic_grade, grade_stats, TASK_SPECS
from server.baseline.random_agent import random_policy, block_all_policy
from server.baseline.heuristic_agent import heuristic_policy
from server.utils.data_loader import TrafficGenerator, FEATURE_ORDER
from server.utils.threat_engine import ThreatEngine


# ===================================================================
# Helpers
# ===================================================================

OK = "[OK]"
FAIL = "[FAIL]"
WARN = "[WARN]"


def allow_all_policy(env, session_ids):
    return {sid: 0 for sid in session_ids}


def separator(title):
    w = 72
    print()
    print("=" * w)
    print("  " + title)
    print("=" * w)


def subsection(title):
    print("\n  -- {} --".format(title))


# ===================================================================
# Test 1: Parameter configuration validation
# ===================================================================

def check_config_parameters():
    separator("1. TASK CONFIG PARAMETER VALIDATION")
    required_keys = [
        "max_steps", "benign_ratio", "threat_probability", "traffic_lambda",
        "budget", "noise_level", "stealth_multiplier", "session_ttl_benign",
        "session_ttl_malicious", "escalation_rate_mod", "false_flag_prob",
        "burst_prob", "burst_size_mult",
    ]
    all_ok = True
    for task, config in TASK_CONFIGS.items():
        missing = [k for k in required_keys if k not in config]
        if missing:
            print("  {} {}: missing keys {}".format(FAIL, task, missing))
            all_ok = False
        else:
            print("  {} {}: all {} parameters present".format(OK, task, len(required_keys)))

        # Validate ranges
        assert 0.0 <= config["noise_level"] <= 0.5, "noise_level out of range for {}".format(task)
        assert 0.5 <= config["stealth_multiplier"] <= 3.0, "stealth_multiplier out of range for {}".format(task)
        assert 1 <= config["session_ttl_benign"] <= 10, "session_ttl_benign out of range for {}".format(task)
        assert 1 <= config["session_ttl_malicious"] <= 10, "session_ttl_malicious out of range for {}".format(task)
        assert 0.5 <= config["escalation_rate_mod"] <= 3.0, "escalation_rate_mod out of range for {}".format(task)
        assert 0.0 <= config["false_flag_prob"] <= 0.5, "false_flag_prob out of range for {}".format(task)
        assert 0.0 <= config["burst_prob"] <= 0.5, "burst_prob out of range for {}".format(task)
        assert 1.0 <= config["burst_size_mult"] <= 5.0, "burst_size_mult out of range for {}".format(task)

    # Check difficulty progression
    subsection("Difficulty progression check")
    tasks = ["easy", "medium", "hard"]
    for param in ["noise_level", "stealth_multiplier", "escalation_rate_mod",
                   "false_flag_prob", "burst_prob", "burst_size_mult"]:
        values = [TASK_CONFIGS[t][param] for t in tasks]
        monotonic = all(values[i] <= values[i + 1] for i in range(len(values) - 1))
        status = OK if monotonic else WARN
        print("  {} {:25s}: easy={:.3f}  med={:.3f}  hard={:.3f}".format(
            status, param, values[0], values[1], values[2]))

    return all_ok


# ===================================================================
# Test 2: Run all policies across all tasks
# ===================================================================

def run_full_evaluation():
    separator("2. POLICY EVALUATION ACROSS ALL TASKS")
    policies = {
        "random":      random_policy(seed=42),
        "block_all":   block_all_policy,
        "allow_all":   allow_all_policy,
        "heuristic":   heuristic_policy,
    }
    tasks = ["easy", "medium", "hard"]
    results = {}

    for task in tasks:
        subsection("Task: {} (threshold={})".format(task.upper(), TASK_SPECS[task].threshold))
        results[task] = {}

        for pname, policy in policies.items():
            env = FirewallEnvironment(seed=TASK_SPECS[task].seed)
            t0 = time.time()
            grade = run_deterministic_grade(env, task=task, policy=policy)
            elapsed = time.time() - t0
            stats = env.get_network_stats()

            results[task][pname] = {
                "score": grade["score"],
                "passed": grade["passed"],
                "stats": stats,
                "elapsed": elapsed,
            }

            passed_str = "PASS " + OK if grade["passed"] else "FAIL " + FAIL
            print("  {:12s}  score={:.4f}  {}  det={:.3f}  fp={:.3f}  eff={:.3f}  ({:.2f}s)".format(
                pname, grade["score"], passed_str,
                stats["detection_rate"], stats["false_positive_rate"],
                stats["efficiency"], elapsed))

            # Print extended metrics
            ff_seen = stats.get("false_flags_seen", 0)
            ff_acc = stats.get("false_flag_accuracy", 0)
            stealth_seen = stats.get("stealth_attacks_seen", 0)
            stealth_det = stats.get("stealth_detection_rate", 0)
            bursts = stats.get("burst_ticks", 0)
            print("  {:12s}  false_flags={} (acc={:.3f})  stealth={} (det={:.3f})  bursts={}".format(
                "", ff_seen, ff_acc, stealth_seen, stealth_det, bursts))

    return results


# ===================================================================
# Test 3: Policy ordering validation
# ===================================================================

def check_policy_ordering(results):
    separator("3. POLICY ORDERING VALIDATION")
    all_ok = True

    for task in ["easy", "medium", "hard"]:
        h_score = results[task]["heuristic"]["score"]
        r_score = results[task]["random"]["score"]
        b_score = results[task]["block_all"]["score"]
        a_score = results[task]["allow_all"]["score"]

        h_vs_r = OK if h_score > r_score else FAIL
        h_vs_b = OK if h_score > b_score else FAIL
        h_vs_a = OK if h_score > a_score else FAIL

        print("  {:8s}  heuristic({:.4f}) > random({:.4f})     {}".format(task, h_score, r_score, h_vs_r))
        print("  {:8s}  heuristic({:.4f}) > block_all({:.4f})   {}".format("", h_score, b_score, h_vs_b))
        print("  {:8s}  heuristic({:.4f}) > allow_all({:.4f})   {}".format("", h_score, a_score, h_vs_a))

        if not (h_score > r_score and h_score > b_score):
            all_ok = False

    return all_ok


# ===================================================================
# Test 4: Feature separation analysis
# ===================================================================

def check_feature_separation():
    separator("4. FEATURE SEPARATION (BENIGN vs MALICIOUS)")
    gen = TrafficGenerator(seed=777)

    benign_vecs = []
    for _ in range(200):
        s = gen.generate_benign_sessions(tick=0, count=1)[0]
        benign_vecs.append(gen.to_observation_vector(s))

    mal_vecs = []
    for scenario in ["port_scan_exploit_c2", "credential_stuffing_lateral",
                      "supply_chain_compromise", "low_and_slow_apt", "ddos_amplification"]:
        for phase in range(4):
            for _ in range(10):
                s = gen.generate_malicious_sessions(
                    tick=0, count=1, attack_phase=phase, scenario=scenario,
                )[0]
                mal_vecs.append(gen.to_observation_vector(s))

    benign_arr = np.array(benign_vecs)
    mal_arr = np.array(mal_vecs)

    mean_diff = np.abs(benign_arr.mean(axis=0) - mal_arr.mean(axis=0))
    top_features = np.argsort(mean_diff)[::-1]

    print("  Feature separation (|mean_diff|):")
    print("  {:30s}  {:>10s}  {:>12s}  {:>8s}  {:>8s}".format(
        "Feature", "Benign u", "Malicious u", "|D|", "Signal"))
    print("  " + "-" * 75)
    significant_count = 0
    for idx in top_features:
        name = FEATURE_ORDER[idx]
        b_mean = benign_arr[:, idx].mean()
        m_mean = mal_arr[:, idx].mean()
        delta = mean_diff[idx]
        signal = "STRONG" if delta > 0.15 else ("MODERATE" if delta > 0.08 else "WEAK")
        if delta > 0.08:
            significant_count += 1
        print("  {:30s}  {:10.4f}  {:12.4f}  {:8.4f}  {:>8s}".format(
            name, b_mean, m_mean, delta, signal))

    print("\n  Significant features (|D| > 0.08): {}/22".format(significant_count))
    ok = significant_count >= 5
    print("  {} Minimum 5 significant features: {}".format(OK if ok else FAIL, significant_count))
    return ok


# ===================================================================
# Test 5: Noise effect validation
# ===================================================================

def check_noise_effect():
    separator("5. OBSERVATION NOISE VALIDATION")
    all_ok = True

    for task in ["easy", "medium", "hard"]:
        noise_level = TASK_CONFIGS[task]["noise_level"]
        env = FirewallEnvironment(seed=42)
        env.reset(task=task, seed=42)

        if not env._session_queue:
            print("  {} No sessions for {}".format(WARN, task))
            continue

        sid = env._session_queue[0]
        session = env.pending_sessions.get(sid) or env.inspected_sessions.get(sid)
        clean_obs = env.generator.to_observation_vector(session)
        noisy_obs = env._current_observation()

        diff = np.array(clean_obs) - np.array(noisy_obs)
        l2 = np.linalg.norm(diff)
        max_diff = np.max(np.abs(diff))

        ok = True
        if noise_level > 0.0:
            ok = l2 > 0.0
        all_ok = all_ok and ok

        status = OK if ok else FAIL
        print("  {} {:8s}: noise_s={:.3f}  L2_diff={:.4f}  max_abs_diff={:.4f}".format(
            status, task, noise_level, l2, max_diff))

    return all_ok


# ===================================================================
# Test 6: Burst & False flag effect
# ===================================================================

def check_burst_and_false_flags():
    separator("6. BURST & FALSE FLAG EFFECT VALIDATION")
    all_ok = True

    for task in ["medium", "hard"]:
        env = FirewallEnvironment(seed=42)
        grade = run_deterministic_grade(env, task=task, policy=heuristic_policy)
        stats = env.get_network_stats()

        ff_prob = TASK_CONFIGS[task]["false_flag_prob"]
        burst_prob = TASK_CONFIGS[task]["burst_prob"]
        ff_seen = stats.get("false_flags_seen", 0)
        bursts = stats.get("burst_ticks", 0)

        ff_ok = ff_seen > 0 if ff_prob > 0 else True
        burst_ok = bursts > 0 if burst_prob > 0 else True
        ok = ff_ok and burst_ok

        all_ok = all_ok and ok
        status = OK if ok else FAIL
        print("  {} {:8s}: false_flag_prob={:.2f} -> seen={:3d}  |  burst_prob={:.2f} -> ticks={:3d}".format(
            status, task, ff_prob, ff_seen, burst_prob, bursts))

    return all_ok


# ===================================================================
# Test 7: Escalation rate effect
# ===================================================================

def check_escalation_effect():
    separator("7. ESCALATION RATE MODIFIER VALIDATION")
    gen = TrafficGenerator(seed=100)

    results = {}
    # Use only 10 ticks to avoid phase saturation (max phase = 3)
    # which would mask the escalation rate differences
    for mod_label, mod_val in [("normal (1.0)", 1.0), ("fast (1.5)", 1.5), ("very_fast (2.0)", 2.0)]:
        engine = ThreatEngine(seed=200)
        total_phases = 0
        n_attackers = 0

        for _ in range(50):
            engine.maybe_spawn_attacker(0.5)

        for tick in range(10):
            engine.generate_attack_sessions(
                tick=tick, generator=gen,
                blocked_attackers=set(),
                escalation_rate_mod=mod_val,
            )

        for a in list(engine._active_attackers.values()) + engine._dead_attackers:
            total_phases += a.phase
            n_attackers += 1

        avg_phase = total_phases / max(n_attackers, 1)
        results[mod_label] = avg_phase
        print("  escalation_mod={:20s}  attackers={:3d}  avg_phase={:.2f}".format(
            mod_label, n_attackers, avg_phase))

    labels = list(results.keys())
    ok = results[labels[0]] <= results[labels[1]] <= results[labels[2]]
    status = OK if ok else WARN
    print("  {} Phase progression monotonic with escalation rate".format(status))
    return ok


# ===================================================================
# Test 8: Budget & efficiency tracking
# ===================================================================

def check_budget_invariants():
    separator("8. BUDGET & EFFICIENCY INVARIANTS")
    all_ok = True

    for task in ["easy", "medium", "hard"]:
        env = FirewallEnvironment(seed=42)
        env.reset(task=task, seed=42)
        initial_budget = env.initial_budget

        done = False
        while not done:
            sids = list(env.inspected_sessions.keys()) + list(env.pending_sessions.keys())
            actions = heuristic_policy(env, sids)
            resp = env.step(actions)
            done = resp["done"]

        stats = env.get_network_stats()
        budget_ok = 0.0 <= env.budget_remaining <= initial_budget
        eff_ok = 0.0 <= stats["efficiency"] <= 1.0
        det_ok = 0.0 <= stats["detection_rate"] <= 1.0
        fp_ok = 0.0 <= stats["false_positive_rate"] <= 1.0

        ok = budget_ok and eff_ok and det_ok and fp_ok
        all_ok = all_ok and ok
        status = OK if ok else FAIL
        print("  {} {:8s}: budget={:.1f}/{:.1f}  eff={:.4f}  det={:.4f}  fp={:.4f}".format(
            status, task, env.budget_remaining, initial_budget,
            stats["efficiency"], stats["detection_rate"], stats["false_positive_rate"]))

    return all_ok


# ===================================================================
# Test 9: Determinism check
# ===================================================================

def check_determinism():
    separator("9. DETERMINISM VALIDATION")
    all_ok = True

    for task in ["easy", "medium", "hard"]:
        env1 = FirewallEnvironment(seed=42)
        env2 = FirewallEnvironment(seed=42)
        policy = random_policy(seed=99)
        policy2 = random_policy(seed=99)

        g1 = run_deterministic_grade(env1, task=task, policy=policy)
        g2 = run_deterministic_grade(env2, task=task, policy=policy2)

        ok = g1["score"] == g2["score"]
        all_ok = all_ok and ok
        status = OK if ok else FAIL
        print("  {} {:8s}: run1={:.6f}  run2={:.6f}".format(status, task, g1["score"], g2["score"]))

    return all_ok


# ===================================================================
# Main
# ===================================================================

def main():
    print()
    print("+" + "=" * 70 + "+")
    print("|  AI FIREWALL ENVIRONMENT -- ACCURACY & PARAMETER CHECKER           |")
    print("+" + "=" * 70 + "+")

    t0 = time.time()
    checks = {}

    checks["config_params"] = check_config_parameters()
    results = run_full_evaluation()
    checks["policy_ordering"] = check_policy_ordering(results)
    checks["feature_separation"] = check_feature_separation()
    checks["noise_effect"] = check_noise_effect()
    checks["burst_false_flags"] = check_burst_and_false_flags()
    checks["escalation_effect"] = check_escalation_effect()
    checks["budget_invariants"] = check_budget_invariants()
    checks["determinism"] = check_determinism()

    # -- Summary --
    separator("SUMMARY")
    total_elapsed = time.time() - t0
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    for name, ok in checks.items():
        status = OK + " PASS" if ok else FAIL + " FAIL"
        print("  {}  {}".format(status, name))

    print("\n  {}/{} checks passed  ({:.1f}s total)".format(passed, total, total_elapsed))

    if passed == total:
        print("\n  ALL CHECKS PASSED -- Environment is accurate and well-calibrated!")
    else:
        print("\n  {} check(s) need attention".format(total - passed))

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
