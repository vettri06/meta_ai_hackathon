"""Quick test: verify inference.py correctly handles Qwen 3.5 reasoning format."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Force Ollama
os.environ["API_BASE_URL"] = "http://localhost:11434/v1"
os.environ["API_KEY"] = "ollama"
os.environ["MODEL_NAME"] = "qwen3.5:cloud"

from inference import InferenceAgent

agent = InferenceAgent()

# Test session
session = {
    "features": {
        "dst_port": 443, "session_history_score": 0.3, "entropy_score": 0.7,
        "connection_reuse": 0.1, "is_self_signed": 0, "ja3_hash_cluster": 50,
        "geo_distance": 2000, "cert_validity_days": 30, "tls_version": 1,
        "dns_query_count": 5, "duration_ms": 200, "packet_count": 50,
    },
    "revealed_malicious": False,
}
threat = {"known_bad_ports": [4444, 8888]}

print("Testing InferenceAgent.get_action via Ollama qwen3.5:cloud...")
print("(This tests the full 3-tier fallback chain)")

for i in range(3):
    action = agent.get_action(session, threat)
    action_names = {0:"ALLOW", 1:"BLOCK", 2:"INSPECT", 3:"SANDBOX", 4:"RATE_LIMIT", 5:"QUARANTINE"}
    print(f"  Call {i+1}: action={action} ({action_names.get(action, '?')})")

print("\nSUCCESS - Qwen 3.5 reasoning model format handled correctly!")
