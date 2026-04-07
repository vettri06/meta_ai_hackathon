from __future__ import annotations

import pytest

from server.baseline.heuristic_agent import heuristic_policy
from server.baseline.random_agent import random_policy
from server.firewall_environment import FirewallEnvironment


@pytest.fixture
def env_easy() -> FirewallEnvironment:
    env = FirewallEnvironment(seed=101)
    env.reset(task="easy", seed=101)
    return env


@pytest.fixture
def env_medium() -> FirewallEnvironment:
    env = FirewallEnvironment(seed=202)
    env.reset(task="medium", seed=202)
    return env


@pytest.fixture
def env_hard() -> FirewallEnvironment:
    env = FirewallEnvironment(seed=303)
    env.reset(task="hard", seed=303)
    return env


@pytest.fixture
def random_agent_policy():
    return random_policy(seed=9)


@pytest.fixture
def heuristic_agent_policy():
    return heuristic_policy
