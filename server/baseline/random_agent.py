"""Random baseline agent for the Adaptive AI Firewall environment."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np


def random_policy(seed: int = 42) -> Callable:
    """Return a random policy function seeded for reproducibility."""
    rng = np.random.default_rng(seed)

    def _policy(env, session_ids: List[str]) -> Dict[str, int]:
        return {sid: int(rng.integers(0, 6)) for sid in session_ids}

    return _policy


def block_all_policy(env, session_ids: List[str]) -> Dict[str, int]:
    """Block every session — useful as a degenerate baseline."""
    return {sid: 1 for sid in session_ids}
