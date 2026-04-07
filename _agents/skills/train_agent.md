# Train Agent

1. Start with `step_single` mode to get fixed-shape RL training (`Discrete(6)`).
2. Use medium task for initial optimization stability; then curriculum to hard.
3. Track reward decomposition (security, availability, efficiency, timeliness) each epoch.
4. Include inspected-session follow-up actions in policy design.
5. Validate every checkpoint with deterministic graders on all tasks.
6. Promote models only if:
   - heuristic-level or better easy score
   - non-zero detection and acceptable false-positive handling
