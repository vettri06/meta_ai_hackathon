# Understand Environment

1. Read `server/firewall_environment.py` for:
   - multi-session mode (`step`)
   - single-session mode (`step_single`)
   - inspect follow-up lifecycle and budget mechanics
2. Read `server/traffic_generator.py` for:
   - feature order and normalization
   - scenario- and phase-specific malicious profiles
3. Read `server/threat_engine.py` for:
   - attacker lifecycle and adaptation
   - attacker outcomes (`active`, `stopped`, `succeeded`)
4. Read `server/reward_engine.py` for:
   - reward weights and anti-degeneracy design
5. Read `server/graders.py` for:
   - deterministic seeds
   - thresholds and pass constraints
