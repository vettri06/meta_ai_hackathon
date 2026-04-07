# Evaluate Agent

1. Run deterministic evaluation:
   - `python -m adaptive_firewall_env.baseline.evaluate`
2. Compare policy against four references:
   - random
   - heuristic
   - block-all
   - allow-all
3. Confirm pass criteria includes both:
   - weighted score threshold
   - pass constraints (`min_detection_rate`, `min_fp_complement`)
4. Inspect per-task metrics:
   - detection rate
   - false-positive complement
   - efficiency
   - cascade prevention
