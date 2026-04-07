from server.baseline.heuristic_agent import heuristic_policy
from server.baseline.random_agent import random_policy
from server.firewall_environment import FirewallEnvironment
from server.graders import run_deterministic_grade


def always_allow_policy(_, session_ids):
    return {sid: 0 for sid in session_ids}


def always_block_policy(_, session_ids):
    return {sid: 1 for sid in session_ids}


def test_policy_ordering_easy_task():
    env = FirewallEnvironment(seed=77)
    random_score = run_deterministic_grade(env, task="easy", policy=random_policy(seed=7))["score"]
    heuristic_score = run_deterministic_grade(env, task="easy", policy=heuristic_policy)["score"]
    allow_score = run_deterministic_grade(env, task="easy", policy=always_allow_policy)["score"]
    assert heuristic_score >= random_score
    assert heuristic_score >= allow_score


def test_block_all_is_not_best_strategy():
    env = FirewallEnvironment(seed=88)
    for task in ("easy", "medium", "hard"):
        block_score = run_deterministic_grade(env, task=task, policy=always_block_policy)["score"]
        heuristic_score = run_deterministic_grade(env, task=task, policy=heuristic_policy)["score"]
        assert block_score <= heuristic_score
