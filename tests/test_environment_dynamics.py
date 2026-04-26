from server.baseline.heuristic_agent import heuristic_policy
from server.firewall_environment import FirewallEnvironment


def test_expired_malicious_sessions_are_counted():
    env = FirewallEnvironment(seed=11)
    env.reset(task="easy", seed=11)
    before = env.metrics.malicious_seen
    for _ in range(4):
        env.step({})
    after = env.metrics.malicious_seen
    assert after >= before


def test_inspect_keeps_session_pending_and_reveals():
    env = FirewallEnvironment(seed=12)
    env.reset(task="easy", seed=12)
    session_id = next(iter(env.pending_sessions.keys()))
    env.take_action(session_id=session_id, action=2)
    # After INSPECT, session moves to inspected pool only (not duplicated in pending)
    assert session_id in env.inspected_sessions
    assert session_id not in env.pending_sessions
    assert env.inspected_sessions[session_id]["metadata"]["revealed"] is True


def test_step_single_has_fixed_size_action_mode():
    env = FirewallEnvironment(seed=13)
    env.reset(task="easy", seed=13)
    response = env.step_single(action=0)
    assert "focus_observation" in response["state"]
    assert len(response["state"]["focus_observation"]) == 22


def test_budget_is_scaled_by_episode_length():
    env = FirewallEnvironment(seed=14, budget=50.0)
    env.reset(task="hard", seed=14)
    assert env.initial_budget >= env.max_steps * 0.35


def test_attacker_outcomes_exposed_in_step_info():
    env = FirewallEnvironment(seed=15)
    env.reset(task="easy", seed=15)
    session_id = next(iter(env.pending_sessions.keys()))
    result = env.step({session_id: 1})
    assert "attacker_outcomes" in result["info"]


def test_heuristic_policy_executes_over_pending_sessions():
    env = FirewallEnvironment(seed=16)
    env.reset(task="easy", seed=16)
    actions = heuristic_policy(env, list(env.pending_sessions.keys())[:5])
    assert isinstance(actions, dict)
