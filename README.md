---
title: AI Firewall OpenEnv
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
app_port: 7860
---

# 🛡️ AI Firewall OpenEnv

A production-grade AI-driven adaptive firewall simulation for automated threat detection in encrypted network traffic.

## 📖 Problem Description
Encrypted traffic poses a challenge for traditional firewalls. This project uses AI agents to make real-time decisions (ALLOW, BLOCK, etc.) based on session metadata alone, balancing security with network performance.

## 🎮 Tasks
- **🟢 Easy (Perimeter Defense)**: Clear attack patterns for initial testing.
- **🟡 Medium (Mixed Threat Landscape)**: Multi-stage attacks with ambiguous traffic signals.
- **🔴 Hard (Advanced Persistent Threat)**: Stealthy, low-signal APT scenarios.

## 🧠 Environment Specs
- **Observation Space**: Box(22,) - Normalized features including JA3 fingerprints, entropy, geo-distance, and session history.
- **Action Space**: Discrete(6)
  - 0: ALLOW
  - 1: BLOCK
  - 2: INSPECT
  - 3: SANDBOX
  - 4: RATE_LIMIT
  - 5: QUARANTINE

## 📊 Reward Logic
Rewards are multi-objective:
- **Correct Block**: +1.0
- **False Positive**: -1.2 (Strong penalty)
- **Missed Attack**: -2.0 (Critical failure)
- **Correct Allow**: +0.25 (Efficiency bonus)
- **Inspect**: Dynamic cost/benefit based on revealed status.

## 🚀 Setup & Usage
### **Prerequisites**
- Docker installed
- Python 3.11+
- API Keys for OpenAI/OpenRouter (optional for LLM agent)

### **Local Execution**
1. **Configure Keys**: `cp .env.example .env` and add your keys.
2. **Run Inference**: `python inference.py --task easy`
3. **Validate**: `bash scripts/validate-submission.sh <ping_url>`

### **Docker Deployment**
```bash
docker build -t ai-firewall .
docker run -p 7860:7860 ai-firewall
```

## 🏗️ Project Structure
- `env/`: Core firewall environment (reset, step, state).
- `grader/`: Scoring and grading logic.
- `utils/`: Traffic simulation and reward engines.
- `inference.py`: LLM-based inference script.
- `openenv.yaml`: Metadata for OpenEnv.
