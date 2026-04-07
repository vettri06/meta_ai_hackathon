# Action Space

## Discrete Actions

| ID | Action | Typical Use | Cost Type |
|---|---|---|---|
| 0 | `ALLOW` | pass low-risk traffic | none |
| 1 | `BLOCK` | immediate deny for high-confidence malicious sessions | low |
| 2 | `INSPECT` | collect additional evidence before terminal decision | medium |
| 3 | `SANDBOX` | isolate unknown/high-risk behavior | high |
| 4 | `RATE_LIMIT` | mitigate volumetric or burst anomalies | low-medium |
| 5 | `QUARANTINE` | isolate source identity while preserving observation | medium |

Costs are computed in `reward_engine.py` as latency + compute.

## Decision Pattern

1. If confidence is high and malicious indicators are strong: `BLOCK` / `QUARANTINE`.
2. If confidence is low but suspicious: `INSPECT` then follow-up action.
3. If traffic appears benign and reputation is healthy: `ALLOW`.
4. If volumetric anomaly dominates: `RATE_LIMIT` before hard block.

## RL Compatibility

- `action_space` is `Discrete(6)` in single-session mode.
- Multi-session mode applies the same discrete action per session ID in the action map.
