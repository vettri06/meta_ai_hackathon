from server.firewall_environment import FirewallEnvironment
from server.graders import run_deterministic_grade

def new_heuristic_policy(env, session_ids):
    threat_intel = env.get_threat_intelligence()
    known_bad_ports = set(threat_intel.get("known_bad_ports", []))
    actions = {}

    for sid in session_ids:
        try:
            data = env.evaluate_session(sid)
        except KeyError:
            actions[sid] = 0
            continue

        features = data.get("features", {})
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

        if ja3 >= 130:
            actions[sid] = 1
        elif dst_port in known_bad_ports and reuse < 0.4:
            actions[sid] = 1
        elif self_signed == 1 and reuse < 0.5:
            actions[sid] = 5
        elif reuse < 0.45 and dns_q >= 3:
            actions[sid] = 1
        elif dur < 100.0 and pkts > 50 and reuse < 0.2:
            actions[sid] = 4
        elif reuse < 0.55 and dns_q >= 3:
            actions[sid] = 2
        elif cert_valid < 250.0 and reuse < 0.6:
            actions[sid] = 2
        elif entropy > 0.55 and reuse < 0.5:
            actions[sid] = 2
        else:
            actions[sid] = 0

    return actions

for task in ['easy', 'medium', 'hard']:
    env = FirewallEnvironment(seed=303)
    res = run_deterministic_grade(env, task, new_heuristic_policy)
    print(f"{task}: score={res['score']:.4f} det={res['breakdown']['detection_rate']:.4f} fp_comp={res['breakdown']['fp_complement']:.4f} eff={res['breakdown']['efficiency']:.4f}")
