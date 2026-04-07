"""Heuristic baseline agent for the Adaptive AI Firewall environment.

Uses the same 8-rule heuristic as inference.py for deterministic testing.
"""
from __future__ import annotations

from typing import Dict, List


def heuristic_policy(env, session_ids: List[str]) -> Dict[str, int]:
    """Rule-based policy using session features and threat intelligence."""
    threat_intel = env.get_threat_intelligence()
    known_bad_ports = set(threat_intel.get("known_bad_ports", []))
    actions: Dict[str, int] = {}

    for sid in session_ids:
        try:
            data = env.evaluate_session(sid)
        except KeyError:
            actions[sid] = 0
            continue

        features = data.get("features", {})

        # If already revealed as malicious, block immediately
        if data.get("revealed_malicious") is True:
            actions[sid] = 1
            continue

        dst_port = int(features.get("dst_port", 0))
        history = float(features.get("session_history_score", 1.0))
        entropy = float(features.get("entropy_score", 0.0))
        reuse = float(features.get("connection_reuse", 1.0))
        self_signed = int(features.get("is_self_signed", 0))
        ja3 = int(features.get("ja3_hash_cluster", 0))
        geo = float(features.get("geo_distance", 0.0))
        cert_valid = float(features.get("cert_validity_days", 999.0))
        tls_ver = int(features.get("tls_version", 1))
        dns_q = int(features.get("dns_query_count", 0))
        dur = float(features.get("duration_ms", 500.0))
        pkts = int(features.get("packet_count", 10))

        if dst_port in known_bad_ports and history < 0.50:
            actions[sid] = 1
        elif self_signed == 1 and history < 0.45:
            actions[sid] = 5
        elif entropy > 0.55 and reuse < 0.25:
            actions[sid] = 2
        elif geo > 4000.0 and history < 0.40:
            actions[sid] = 2
        elif ja3 >= 180:
            actions[sid] = 1
        elif dur < 60.0 and pkts > 100:
            actions[sid] = 4
        elif cert_valid < 80.0 and tls_ver == 0:
            actions[sid] = 2
        elif reuse < 0.10 and dns_q >= 4:
            actions[sid] = 2
        else:
            actions[sid] = 0

    return actions
