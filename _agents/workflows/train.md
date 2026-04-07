# Train Workflow

1. Establish reference:
   - run baseline evaluator and record heuristic score per task.
2. Begin in single-session mode (`step_single`) with medium task.
3. Train policy network on normalized 22-dim observations and `Discrete(6)` actions.
4. Include inspect follow-up strategy in action head logic.
5. Evaluate every checkpoint on deterministic seeds.
6. Promote model only if:
   - easy and medium pass constraints satisfied
   - hard score improves over random baseline
   - no degeneration to block-all or allow-all behavior.
