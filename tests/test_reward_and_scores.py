from server.firewall_environment import FirewallEnvironment
from server.graders import grade_stats
from server.utils.reward_engine import RewardEngine


def test_grade_score_bounds():
    stats = {
        "detection_rate": 0.5,
        "false_positive_rate": 0.1,
        "efficiency": 0.8,
        "early_detection_bonus": 0.7,
        "cascade_prevention": 0.6,
    }
    for task in ("easy", "medium", "hard"):
        score = grade_stats(task, stats)["score"]
        assert 0.0 <= score <= 1.0


def test_reward_range_is_reasonable():
    engine = RewardEngine()
    samples = [
        engine.reward(action=0, is_malicious=False, budget_remaining=100.0, attack_phase=0)[0],
        engine.reward(action=1, is_malicious=False, budget_remaining=100.0, attack_phase=0)[0],
        engine.reward(action=1, is_malicious=True, budget_remaining=100.0, attack_phase=1)[0],
        engine.reward(action=0, is_malicious=True, budget_remaining=100.0, attack_phase=3)[0],
    ]
    assert min(samples) > -2.5
    assert max(samples) < 2.5


def test_efficiency_is_non_zero_after_episode():
    env = FirewallEnvironment(seed=66)
    env.reset(task="medium", seed=66)
    done = False
    while not done:
        response = env.step({})
        done = response["done"]
    stats = env.get_network_stats()
    assert stats["efficiency"] > 0.0
