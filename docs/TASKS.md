# Tasks

## Difficulty Levels

| Task | Steps | Threshold | Pass Constraints |
|---|---:|---:|---|
| Easy | 200 | 0.70 | detection ≥ 0.35 and fp_complement ≥ 0.65 |
| Medium | 500 | 0.50 | detection ≥ 0.35 and fp_complement ≥ 0.60 |
| Hard | 1000 | 0.45 | detection ≥ 0.35 and fp_complement ≥ 0.55 |

## Why Constraints Exist

Weighted scores alone can be gamed by degenerate policies:
- `allow_all` inflates availability/efficiency.
- `block_all` inflates detection.

The pass constraints ensure any passing policy must satisfy both:
1. meaningful threat detection,
2. acceptable benign-traffic handling.

Task scoring logic is implemented in `server/graders.py`.
