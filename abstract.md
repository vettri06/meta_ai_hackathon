# Project Abstract: Adaptive AI Firewall Simulation

## 1. Overview
The **Adaptive AI Firewall** is a highly sophisticated, reinforcement learning (RL) compatible cybersecurity simulation platform. It bridges the gap between static rule-based security systems and autonomous, AI-driven threat mitigation. By generating realistic, 22-dimensional encrypted network traffic streams, the platform evaluates the ability of Large Language Models (LLMs) and heuristic agents to accurately classify, inspect, and quarantine threats in real-time.

## 2. Core Architecture & Algorithms

The system is built upon several advanced algorithms working in tandem:

### A. The Defense Agents
1. **The Autonomous AI Agent (Transformer LLM)**
   - **Algorithm**: Zero-Shot Contextual Classification using `qwen3.5` (via Ollama API).
   - **Implementation**: The LLM is fed a sanitized JSON representation of a network session alongside strict decision-making rules. It must reason and output a discrete action (0-5).
   - **Actions**: `ALLOW`, `BLOCK`, `INSPECT`, `SANDBOX`, `RATE_LIMIT`, `QUARANTINE`.
2. **The Baseline Heuristic Agent**
   - **Algorithm**: Deterministic Decision Tree.
   - **Implementation**: A fast, rule-based tier-1 defense system that evaluates hard thresholds (e.g., `ja3_hash_cluster >= 130` or `entropy > 0.55`) to serve as a performance benchmark against the LLM.

### B. The Environment & Traffic Generation
1. **Traffic Spawning (Poisson Point Process)**
   - **Algorithm**: Poisson distributions govern the `traffic_lambda` to simulate realistic, bursty network behavior (inter-arrival mean, jitter, bursts).
   - **Features**: Generates 22 distinct metadata features simulating *encrypted* traffic (no payload inspection). Features are grouped into Volume/Timing, Network Metadata, TLS/Certificates, and Behavioral Context (e.g., Geo-distance, Connection Reuse).
2. **Reinforcement Learning Core (OpenEnv)**
   - **Implementation**: The environment (`FirewallEnvironment`) exposes standard RL methods:
     - `reset()`: Initializes time, budget, and queues.
     - `state()`: Returns the current network snapshot (focus session, pending queues, remaining compute budget).
     - `step(action)`: Applies the AI's action, calculates multi-objective reward/cost, ages the session TTLs, and advances time.

### C. Self-Play & Adaptive Training
1. **Automatic Domain Randomization (ADR)**
   - **Algorithm**: The Curriculum Engine dynamically shifts the environment parameters based on the agent's real-time weakness profile.
   - **Implementation**: If the agent masters basic detection, the ADR increases the `stealth_multiplier`, injects higher `noise_level` into features, or increases `false_flag_prob` (benign traffic designed to trigger false positives).
2. **Logistic Elo Rating System**
   - **Algorithm**: Logistic curve (K=32 factor).
   - **Implementation**: Because the environment is non-stationary (it gets harder as the agent gets better), absolute scores are misleading. The system uses an Elo rating algorithm to track the true skill growth of the AI against the shifting environment difficulty.

## 3. Advanced Diagnostic & UI Systems

1. **Multi-Objective Grading Engine**
   - The environment tracks complex metrics beyond simple accuracy, including:
     - **Compute Cost**: Actions like `INSPECT` drain the budget.
     - **Cascade Failures**: Penalties for letting threats linger too long in the network.
     - **Stealth & False Flag Accuracy**: Measuring resistance to adversarial deception.
2. **Automated Diagnostic Pipelines**
   - Generates high-resolution, multi-panel PNG graphs mapping absolute training loss, Elo progression, difficulty-normalized rewards, and confusion matrices after every training run.
3. **Real-Time 3D Visualizer**
   - A fully responsive, glassmorphic Web UI built with HTML/JS and CSS3 Isometric Transforms. It reads the raw output JSON streams and animates the network traffic—showing packets shattering on blocks, pausing for inspection, or entering containment grids based on the AI's real-world decisions.

## 4. Conclusion
This project successfully demonstrates a cutting-edge pipeline for training, validating, and visualizing AI cybersecurity agents. By combining ADR, LLM reasoning, and realistic Poisson-based traffic simulation, it serves as a robust foundation for next-generation, autonomous network defense research.
