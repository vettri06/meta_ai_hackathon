# Threat Models

## Scenario Catalog

| Scenario | Early Phase | Mid Phase | Late Phase |
|---|---|---|---|
| `port_scan_exploit_c2` | rapid probing | exploit delivery | command/control + exfil |
| `credential_stuffing_lateral` | auth pressure | lateral movement | persistence |
| `supply_chain_compromise` | stealth foothold | trusted-channel abuse | disguised exfiltration |
| `low_and_slow_apt` | sparse reconnaissance | long dwell C2 | slow extraction |
| `ddos_amplification` | reflection probes | traffic amplification | flood stage |

## Adaptation Behavior

- Repeated blocking increases attacker detection count.
- Detected attackers can switch to stealth mode and alter feature distributions.
- Attackers terminate when repeatedly blocked, time out, or complete exfiltration.
- Threat engine exposes per-attacker outcomes (`active`, `stopped`, `succeeded`) for analysis and credit assignment.

Threat generation and lifecycle are implemented in `server/threat_engine.py`.
