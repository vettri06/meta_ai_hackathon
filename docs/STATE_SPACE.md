# State Space

The environment uses a **22-dimensional normalized observation vector** (`[0,1]` per feature).
Order is fixed by `FEATURE_ORDER` in `traffic_generator.py`.

## Feature Groups

| Group | Features | Semantics |
|---|---|---|
| Volume & timing | bytes sent/received, duration, packet count, packet variance, inter-arrival mean/jitter | throughput shape and temporal burstiness |
| Network metadata | src/dst ports, protocol, DNS query count, connection reuse | routing and communication pattern |
| TLS / certificate | TLS version, JA3 cluster, chain length, cert validity, self-signed | encrypted-session trust indicators |
| Behavioral context | geo distance, time of day, session history score, entropy score | reputation and anomaly context |

## Observation Interfaces

- `evaluate_session(session_id)` returns the vector for a given session.
- `state()` returns environment-level counters and selected session IDs.
- `step_single(action)` returns `observation` for the next queued session.

## Normalization Strategy

- Each raw feature is min-max normalized using bounded ranges in `FEATURE_BOUNDS`.
- Outliers are clipped to `[0,1]` after normalization.
- This enables stable neural training across heterogeneous scales (ports, durations, entropy).

## Markov Context Notes

- Single-session mode is designed for fixed-shape RL loops.
- Multi-session mode supports tool-driven decision systems over dynamic queues.
