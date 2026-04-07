"""Network traffic session generator with realistic correlated features.

Each session is a 22-dimensional feature vector representing metadata and
behavioral signals from encrypted traffic (no payload inspection).

Feature groups:
  - Volume & timing: bytes, duration, packet stats, inter-arrival metrics
  - Network metadata: ports, protocol, DNS, connection reuse
  - TLS / certificate: TLS version, JA3 cluster, cert chain, self-signed
  - Behavioral context: geo distance, time of day, reputation, entropy

Benign traffic is drawn from 5 profile archetypes.  Malicious traffic
profiles vary by attack scenario AND kill-chain phase, creating real
distributional differences an RL agent can learn to exploit.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import math

import numpy as np


FEATURE_ORDER = [
    "bytes_sent",
    "bytes_received",
    "duration_ms",
    "packet_count",
    "avg_packet_size",
    "packet_size_variance",
    "inter_arrival_mean",
    "inter_arrival_jitter",
    "src_port",
    "dst_port",
    "protocol",
    "tls_version",
    "ja3_hash_cluster",
    "cert_chain_length",
    "cert_validity_days",
    "is_self_signed",
    "dns_query_count",
    "connection_reuse",
    "geo_distance",
    "time_of_day",
    "session_history_score",
    "entropy_score",
]

# Min/max bounds for normalization (empirically calibrated)
FEATURE_BOUNDS: Dict[str, tuple] = {
    "bytes_sent": (4.0, 14.0),
    "bytes_received": (3.0, 13.0),
    "duration_ms": (20.0, 25000.0),
    "packet_count": (2.0, 1200.0),
    "avg_packet_size": (40.0, 1400.0),
    "packet_size_variance": (5.0, 500.0),
    "inter_arrival_mean": (0.5, 600.0),
    "inter_arrival_jitter": (0.0, 300.0),
    "src_port": (1024.0, 65535.0),
    "dst_port": (1.0, 65535.0),
    "protocol": (0.0, 2.0),
    "tls_version": (0.0, 2.0),
    "ja3_hash_cluster": (0.0, 255.0),
    "cert_chain_length": (0.0, 6.0),
    "cert_validity_days": (1.0, 1200.0),
    "is_self_signed": (0.0, 1.0),
    "dns_query_count": (0.0, 12.0),
    "connection_reuse": (0.0, 1.0),
    "geo_distance": (0.0, 12000.0),
    "time_of_day": (0.0, 1.0),
    "session_history_score": (0.0, 1.0),
    "entropy_score": (0.0, 1.0),
}


@dataclass(frozen=True)
class TrafficProfile:
    name: str
    packet_mean: float
    packet_std_frac: float     # std = mean * frac
    duration_mean: float
    entropy_mean: float
    entropy_std: float
    tls_probability: float
    self_signed_prob: float
    common_ports: List[int]
    connection_reuse_mean: float
    geo_distance_mean: float
    history_score_mean: float
    cert_validity_mean: float
    ja3_cluster_range: tuple = (0, 128)


# ── Benign traffic profiles ─────────────────────────────────────────
BENIGN_PROFILES = [
    TrafficProfile(
        name="WebBrowsing", packet_mean=50.0, packet_std_frac=0.35,
        duration_mean=900.0, entropy_mean=0.32, entropy_std=0.06,
        tls_probability=0.95, self_signed_prob=0.02,
        common_ports=[80, 443], connection_reuse_mean=0.72,
        geo_distance_mean=1400.0, history_score_mean=0.82,
        cert_validity_mean=450.0, ja3_cluster_range=(0, 64),
    ),
    TrafficProfile(
        name="Streaming", packet_mean=800.0, packet_std_frac=0.25,
        duration_mean=18000.0, entropy_mean=0.22, entropy_std=0.04,
        tls_probability=0.99, self_signed_prob=0.01,
        common_ports=[443, 8080], connection_reuse_mean=0.88,
        geo_distance_mean=2200.0, history_score_mean=0.90,
        cert_validity_mean=500.0, ja3_cluster_range=(0, 32),
    ),
    TrafficProfile(
        name="API", packet_mean=25.0, packet_std_frac=0.30,
        duration_mean=350.0, entropy_mean=0.18, entropy_std=0.04,
        tls_probability=0.98, self_signed_prob=0.01,
        common_ports=[443, 8443], connection_reuse_mean=0.80,
        geo_distance_mean=1000.0, history_score_mean=0.85,
        cert_validity_mean=500.0, ja3_cluster_range=(0, 48),
    ),
    TrafficProfile(
        name="IoT", packet_mean=10.0, packet_std_frac=0.40,
        duration_mean=1500.0, entropy_mean=0.38, entropy_std=0.07,
        tls_probability=0.30, self_signed_prob=0.08,
        common_ports=[1883, 5683, 8883], connection_reuse_mean=0.55,
        geo_distance_mean=800.0, history_score_mean=0.70,
        cert_validity_mean=300.0, ja3_cluster_range=(80, 128),
    ),
    TrafficProfile(
        name="Enterprise", packet_mean=120.0, packet_std_frac=0.35,
        duration_mean=1200.0, entropy_mean=0.28, entropy_std=0.06,
        tls_probability=0.85, self_signed_prob=0.04,
        common_ports=[443, 445, 3389], connection_reuse_mean=0.65,
        geo_distance_mean=500.0, history_score_mean=0.88,
        cert_validity_mean=400.0, ja3_cluster_range=(0, 96),
    ),
]

# ── Malicious traffic profiles per (scenario, phase) ────────────────
# Each scenario has distinct fingerprints making them differentiable
MALICIOUS_PROFILES: Dict[str, Dict[int, TrafficProfile]] = {
    "port_scan_exploit_c2": {
        0: TrafficProfile(
            name="PortScan_Recon", packet_mean=6.0, packet_std_frac=0.5,
            duration_mean=80.0, entropy_mean=0.12, entropy_std=0.04,
            tls_probability=0.05, self_signed_prob=0.60,
            common_ports=[21, 22, 23, 25, 445, 3389, 5900],
            connection_reuse_mean=0.02, geo_distance_mean=5500.0,
            history_score_mean=0.10, cert_validity_mean=60.0,
            ja3_cluster_range=(200, 255),
        ),
        1: TrafficProfile(
            name="PortScan_Exploit", packet_mean=45.0, packet_std_frac=0.4,
            duration_mean=300.0, entropy_mean=0.78, entropy_std=0.06,
            tls_probability=0.40, self_signed_prob=0.45,
            common_ports=[80, 443, 8080, 445],
            connection_reuse_mean=0.08, geo_distance_mean=5200.0,
            history_score_mean=0.12, cert_validity_mean=90.0,
            ja3_cluster_range=(210, 255),
        ),
        2: TrafficProfile(
            name="PortScan_C2", packet_mean=4.0, packet_std_frac=0.6,
            duration_mean=5000.0, entropy_mean=0.55, entropy_std=0.08,
            tls_probability=0.92, self_signed_prob=0.35,
            common_ports=[443, 53, 8443],
            connection_reuse_mean=0.15, geo_distance_mean=6000.0,
            history_score_mean=0.15, cert_validity_mean=45.0,
            ja3_cluster_range=(220, 255),
        ),
        3: TrafficProfile(
            name="PortScan_Exfil", packet_mean=350.0, packet_std_frac=0.3,
            duration_mean=12000.0, entropy_mean=0.88, entropy_std=0.04,
            tls_probability=0.98, self_signed_prob=0.25,
            common_ports=[443, 8443],
            connection_reuse_mean=0.10, geo_distance_mean=6500.0,
            history_score_mean=0.08, cert_validity_mean=30.0,
            ja3_cluster_range=(230, 255),
        ),
    },
    "credential_stuffing_lateral": {
        0: TrafficProfile(
            name="CredStuff_Probe", packet_mean=15.0, packet_std_frac=0.4,
            duration_mean=200.0, entropy_mean=0.42, entropy_std=0.06,
            tls_probability=0.90, self_signed_prob=0.10,
            common_ports=[443, 80, 8443],
            connection_reuse_mean=0.05, geo_distance_mean=3500.0,
            history_score_mean=0.25, cert_validity_mean=300.0,
            ja3_cluster_range=(140, 200),
        ),
        1: TrafficProfile(
            name="CredStuff_Auth", packet_mean=20.0, packet_std_frac=0.35,
            duration_mean=150.0, entropy_mean=0.50, entropy_std=0.07,
            tls_probability=0.95, self_signed_prob=0.08,
            common_ports=[443, 389, 636],
            connection_reuse_mean=0.10, geo_distance_mean=3200.0,
            history_score_mean=0.30, cert_validity_mean=350.0,
            ja3_cluster_range=(150, 210),
        ),
        2: TrafficProfile(
            name="CredStuff_Lateral", packet_mean=30.0, packet_std_frac=0.35,
            duration_mean=500.0, entropy_mean=0.35, entropy_std=0.06,
            tls_probability=0.80, self_signed_prob=0.12,
            common_ports=[445, 3389, 5985, 22],
            connection_reuse_mean=0.20, geo_distance_mean=300.0,
            history_score_mean=0.40, cert_validity_mean=350.0,
            ja3_cluster_range=(160, 220),
        ),
        3: TrafficProfile(
            name="CredStuff_Exfil", packet_mean=200.0, packet_std_frac=0.3,
            duration_mean=8000.0, entropy_mean=0.80, entropy_std=0.05,
            tls_probability=0.98, self_signed_prob=0.15,
            common_ports=[443, 8443],
            connection_reuse_mean=0.12, geo_distance_mean=4000.0,
            history_score_mean=0.18, cert_validity_mean=90.0,
            ja3_cluster_range=(180, 240),
        ),
    },
    "supply_chain_compromise": {
        0: TrafficProfile(
            name="SupplyChain_Init", packet_mean=40.0, packet_std_frac=0.3,
            duration_mean=600.0, entropy_mean=0.30, entropy_std=0.05,
            tls_probability=0.98, self_signed_prob=0.03,
            common_ports=[443, 8443],
            connection_reuse_mean=0.60, geo_distance_mean=1800.0,
            history_score_mean=0.70, cert_validity_mean=380.0,
            ja3_cluster_range=(30, 80),
        ),
        1: TrafficProfile(
            name="SupplyChain_Inject", packet_mean=60.0, packet_std_frac=0.3,
            duration_mean=800.0, entropy_mean=0.40, entropy_std=0.06,
            tls_probability=0.98, self_signed_prob=0.04,
            common_ports=[443, 8443],
            connection_reuse_mean=0.55, geo_distance_mean=2000.0,
            history_score_mean=0.65, cert_validity_mean=350.0,
            ja3_cluster_range=(35, 90),
        ),
        2: TrafficProfile(
            name="SupplyChain_Beacon", packet_mean=8.0, packet_std_frac=0.5,
            duration_mean=3000.0, entropy_mean=0.48, entropy_std=0.07,
            tls_probability=0.99, self_signed_prob=0.05,
            common_ports=[443],
            connection_reuse_mean=0.50, geo_distance_mean=2500.0,
            history_score_mean=0.55, cert_validity_mean=250.0,
            ja3_cluster_range=(40, 100),
        ),
        3: TrafficProfile(
            name="SupplyChain_Exfil", packet_mean=100.0, packet_std_frac=0.3,
            duration_mean=5000.0, entropy_mean=0.60, entropy_std=0.06,
            tls_probability=0.99, self_signed_prob=0.06,
            common_ports=[443, 8443],
            connection_reuse_mean=0.42, geo_distance_mean=3000.0,
            history_score_mean=0.45, cert_validity_mean=200.0,
            ja3_cluster_range=(50, 110),
        ),
    },
    "low_and_slow_apt": {
        0: TrafficProfile(
            name="APT_Recon", packet_mean=12.0, packet_std_frac=0.4,
            duration_mean=400.0, entropy_mean=0.28, entropy_std=0.05,
            tls_probability=0.92, self_signed_prob=0.05,
            common_ports=[443, 80],
            connection_reuse_mean=0.50, geo_distance_mean=2200.0,
            history_score_mean=0.55, cert_validity_mean=320.0,
            ja3_cluster_range=(60, 130),
        ),
        1: TrafficProfile(
            name="APT_Establish", packet_mean=18.0, packet_std_frac=0.35,
            duration_mean=700.0, entropy_mean=0.35, entropy_std=0.06,
            tls_probability=0.95, self_signed_prob=0.07,
            common_ports=[443, 53],
            connection_reuse_mean=0.45, geo_distance_mean=2600.0,
            history_score_mean=0.48, cert_validity_mean=280.0,
            ja3_cluster_range=(70, 140),
        ),
        2: TrafficProfile(
            name="APT_Persist", packet_mean=5.0, packet_std_frac=0.6,
            duration_mean=8000.0, entropy_mean=0.42, entropy_std=0.07,
            tls_probability=0.97, self_signed_prob=0.10,
            common_ports=[443],
            connection_reuse_mean=0.38, geo_distance_mean=3200.0,
            history_score_mean=0.38, cert_validity_mean=200.0,
            ja3_cluster_range=(80, 150),
        ),
        3: TrafficProfile(
            name="APT_Exfil", packet_mean=60.0, packet_std_frac=0.4,
            duration_mean=15000.0, entropy_mean=0.65, entropy_std=0.06,
            tls_probability=0.99, self_signed_prob=0.12,
            common_ports=[443, 8443],
            connection_reuse_mean=0.25, geo_distance_mean=4000.0,
            history_score_mean=0.28, cert_validity_mean=120.0,
            ja3_cluster_range=(90, 160),
        ),
    },
    "ddos_amplification": {
        0: TrafficProfile(
            name="DDoS_Probe", packet_mean=20.0, packet_std_frac=0.5,
            duration_mean=50.0, entropy_mean=0.15, entropy_std=0.04,
            tls_probability=0.10, self_signed_prob=0.30,
            common_ports=[53, 123, 161, 1900],
            connection_reuse_mean=0.02, geo_distance_mean=6000.0,
            history_score_mean=0.08, cert_validity_mean=60.0,
            ja3_cluster_range=(230, 255),
        ),
        1: TrafficProfile(
            name="DDoS_Amplify", packet_mean=500.0, packet_std_frac=0.4,
            duration_mean=30.0, entropy_mean=0.10, entropy_std=0.03,
            tls_probability=0.05, self_signed_prob=0.40,
            common_ports=[53, 123, 161, 1900, 11211],
            connection_reuse_mean=0.01, geo_distance_mean=7000.0,
            history_score_mean=0.05, cert_validity_mean=30.0,
            ja3_cluster_range=(240, 255),
        ),
        2: TrafficProfile(
            name="DDoS_Sustained", packet_mean=900.0, packet_std_frac=0.3,
            duration_mean=20.0, entropy_mean=0.08, entropy_std=0.02,
            tls_probability=0.03, self_signed_prob=0.50,
            common_ports=[53, 123, 80],
            connection_reuse_mean=0.00, geo_distance_mean=8000.0,
            history_score_mean=0.03, cert_validity_mean=20.0,
            ja3_cluster_range=(245, 255),
        ),
        3: TrafficProfile(
            name="DDoS_Peak", packet_mean=1100.0, packet_std_frac=0.25,
            duration_mean=15.0, entropy_mean=0.06, entropy_std=0.02,
            tls_probability=0.02, self_signed_prob=0.55,
            common_ports=[53, 123, 80],
            connection_reuse_mean=0.00, geo_distance_mean=9000.0,
            history_score_mean=0.02, cert_validity_mean=15.0,
            ja3_cluster_range=(248, 255),
        ),
    },
}

# Fallback for unknown scenarios
_DEFAULT_MALICIOUS: Dict[int, TrafficProfile] = MALICIOUS_PROFILES["port_scan_exploit_c2"]

BENIGN_WEIGHTS = np.array([0.34, 0.16, 0.18, 0.12, 0.20])


class TrafficGenerator:
    """Generates correlated network session feature vectors.

    Each session is a dict with 'session_id', 'features' (dict),
    and 'metadata' (malicious flag, attack info, profile name).
    """

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.session_counter = 0

    def generate_benign_sessions(self, tick: int, count: int) -> List[Dict]:
        sessions: List[Dict] = []
        for _ in range(max(0, count)):
            idx = self.rng.choice(len(BENIGN_PROFILES), p=BENIGN_WEIGHTS)
            profile = BENIGN_PROFILES[idx]
            sessions.append(self._build_session(
                profile, tick=tick, malicious=False,
                attack_phase=0, scenario="benign", attacker_id=None,
            ))
        return sessions

    def generate_malicious_sessions(
        self, tick: int, count: int,
        attack_phase: int, scenario: str,
        attacker_id: str | None = None,
    ) -> List[Dict]:
        sessions: List[Dict] = []
        profiles = MALICIOUS_PROFILES.get(scenario, _DEFAULT_MALICIOUS)
        profile = profiles.get(attack_phase, profiles[max(profiles.keys())])
        for _ in range(max(0, count)):
            sessions.append(self._build_session(
                profile, tick=tick, malicious=True,
                attack_phase=attack_phase, scenario=scenario,
                attacker_id=attacker_id,
            ))
        return sessions

    def to_observation_vector(self, session: Dict) -> List[float]:
        """Return normalized [0, 1] feature vector."""
        raw = session["features"]
        normalized = []
        for name in FEATURE_ORDER:
            val = float(raw[name])
            lo, hi = FEATURE_BOUNDS[name]
            normalized.append(max(0.0, min(1.0, (val - lo) / max(hi - lo, 1e-9))))
        return normalized

    def to_raw_vector(self, session: Dict) -> List[float]:
        """Return un-normalized feature vector (for inspection)."""
        return [float(session["features"][name]) for name in FEATURE_ORDER]

    # ── Internal session builder ─────────────────────────────────────

    def _build_session(
        self, profile: TrafficProfile, tick: int,
        malicious: bool, attack_phase: int, scenario: str,
        attacker_id: str | None,
    ) -> Dict:
        self.session_counter += 1
        rng = self.rng

        # --- Volume & timing (correlated cluster) ---
        packet_count = int(max(3, rng.normal(
            profile.packet_mean, profile.packet_mean * profile.packet_std_frac,
        )))
        avg_packet_size = float(max(40.0, rng.normal(560.0, 160.0)))
        # Bytes are correlated with packets and packet size
        bytes_sent = float(max(200.0, packet_count * avg_packet_size * rng.uniform(0.40, 0.85)))
        bytes_received = float(max(100.0, packet_count * avg_packet_size * rng.uniform(0.20, 0.60)))
        duration_ms = float(max(10.0, rng.normal(
            profile.duration_mean, profile.duration_mean * 0.30,
        )))
        # Inter-arrival derived from duration and packet count (correlated)
        inter_arrival_mean = float(duration_ms / max(packet_count, 1))
        inter_arrival_jitter = float(abs(rng.normal(
            inter_arrival_mean * 0.30, inter_arrival_mean * 0.12,
        )))
        packet_size_variance = float(max(5.0, abs(rng.normal(
            180.0 if malicious else 130.0, 60.0,
        ))))

        # --- TLS / certificate (correlated cluster) ---
        tls_enabled = rng.random() < profile.tls_probability
        tls_version = int(rng.choice([1, 2], p=[0.20, 0.80])) if tls_enabled else 0
        # Self-signed correlates with TLS state and profile
        is_self_signed = bool(rng.random() < profile.self_signed_prob) if tls_enabled else False
        cert_chain_length = int(max(0, rng.normal(3.0 if (tls_enabled and not is_self_signed) else 1.0, 0.8)))
        cert_validity_days = float(max(1.0, rng.normal(
            profile.cert_validity_mean, profile.cert_validity_mean * 0.30,
        )))

        # --- Network metadata ---
        dst_port = int(rng.choice(profile.common_ports))
        src_port = int(rng.integers(1024, 65535))
        protocol = int(rng.choice([0, 1, 2], p=[0.50, 0.32, 0.18]))
        dns_query_count = int(max(0, rng.poisson(3 if malicious else 1)))

        # --- Behavioral context (correlated with profile) ---
        connection_reuse = float(np.clip(rng.normal(
            profile.connection_reuse_mean, 0.12,
        ), 0.0, 1.0))
        geo_distance = float(max(0.0, rng.normal(
            profile.geo_distance_mean, profile.geo_distance_mean * 0.25,
        )))
        session_history_score = float(np.clip(rng.normal(
            profile.history_score_mean, 0.10,
        ), 0.0, 1.0))
        entropy_score = float(np.clip(rng.normal(
            profile.entropy_mean, profile.entropy_std,
        ), 0.02, 0.99))
        ja3_lo, ja3_hi = profile.ja3_cluster_range
        ja3_hash_cluster = int(rng.integers(ja3_lo, max(ja3_lo + 1, ja3_hi)))
        time_of_day = float((tick % 1440) / 1440.0)

        features = {
            "bytes_sent": math.log1p(bytes_sent),
            "bytes_received": math.log1p(bytes_received),
            "duration_ms": duration_ms,
            "packet_count": packet_count,
            "avg_packet_size": avg_packet_size,
            "packet_size_variance": packet_size_variance,
            "inter_arrival_mean": inter_arrival_mean,
            "inter_arrival_jitter": inter_arrival_jitter,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "tls_version": tls_version,
            "ja3_hash_cluster": ja3_hash_cluster,
            "cert_chain_length": cert_chain_length,
            "cert_validity_days": cert_validity_days,
            "is_self_signed": int(is_self_signed),
            "dns_query_count": dns_query_count,
            "connection_reuse": connection_reuse,
            "geo_distance": geo_distance,
            "time_of_day": time_of_day,
            "session_history_score": session_history_score,
            "entropy_score": entropy_score,
        }

        # Session TTL: malicious sessions expire faster (pressure to act)
        ttl = 2 if malicious else 3

        return {
            "session_id": f"s-{self.session_counter:07d}",
            "features": features,
            "metadata": {
                "malicious": malicious,
                "attack_phase": attack_phase,
                "scenario": scenario,
                "profile": profile.name,
                "attacker_id": attacker_id,
                "revealed": False,
            },
            "created_tick": tick,
            "expires_tick": tick + ttl,
        }
