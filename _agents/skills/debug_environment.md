# Debug Environment

1. Validate deterministic resets:
   - same seed + same policy must produce same score.
2. Inspect session lifecycle:
   - pending vs inspected pools
   - expiration counts for benign and malicious.
3. Inspect budget dynamics:
   - `budget_remaining`
   - `metrics.total_cost`
   - efficiency in `get_network_stats()`.
4. Diagnose degenerate policy leaks:
   - run block-all / allow-all baselines
   - verify pass constraints reject them.
5. Verify single-session mode:
   - observation size stays fixed (`22`)
   - action range stays `[0..5]`.
