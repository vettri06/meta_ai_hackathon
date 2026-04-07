# 💰 Reward Design — Multi-Objective Optimization

The Adaptive AI Firewall environment uses a sophisticated, weighted reward function designed to drive agent behavior toward a balance of security efficacy, network availability, and resource efficiency.

## 📐 The Reward Equation

The total scalar reward $R$ for any action is calculated as:

$$R = \alpha \cdot R_{\text{security}} + \beta \cdot R_{\text{availability}} + \gamma \cdot R_{\text{efficiency}} + \delta \cdot R_{\text{timeliness}}$$

### **Default Weights**
| Component | Weight | Responsibility |
|---|---|---|
| $\alpha$ | **0.35** | Security Efficacy (Catching threats) |
| $\beta$ | **0.30** | Network Availability (Avoiding False Positives) |
| $\gamma$ | **0.20** | Resource Efficiency (Budget management) |
| $\delta$ | **0.15** | Timeliness (Stopping attacks early) |

---

## 🧩 Reward Components

### **1. Security ($R_{\text{security}}$)**
- **Block Malicious**: $+1.0$ (Successfully stopped a threat).
- **Miss Malicious**: $-2.0$ (Failed to block an attack; high penalty).
- **Inspect Malicious**: $+0.15$ (Correct identification, though not yet stopped).
- **Inspect Benign**: $-0.5$ (Unnecessary inspection).

### **2. Availability ($R_{\text{availability}}$)**
- **Allow Benign**: $+0.25$ (Maintaining network flow).
- **Block Benign (FP)**: $-1.2$ (Significant penalty for disrupting legitimate users).
- **Rate Limit Benign**: $-0.4$ (Milder penalty for "gray" actions).
- **Inspect Benign**: $-0.15$ (Unnecessary latency added).

### **3. Efficiency ($R_{\text{efficiency}}$)**
- **Cost**: Calculated as $\text{latency} + \text{compute}$ for each action.
- **Scaling**: Penalized relative to remaining budget: $R_{\text{efficiency}} = -\frac{\text{cost}}{\max(\text{budget\_remaining}, 0.1)}$.
- This creates **Strategic Pressure**: actions become "more expensive" as the budget depletes.

### **4. Timeliness ($R_{\text{timeliness}}$)**
- **Early Detection**: $+e^{-\text{phase}}$ where `phase` is the attacker's progress in the kill chain (0 to 4).
- **Incentive**: Stopping an attack at Phase 0 is significantly more rewarding than at Phase 3.

---

## 📊 Worked Examples

| Scenario | Action | Security | Availability | Efficiency | Timeliness | **Total Reward** |
|---|---|---|---|---|---|---|
| **Legitimate User** | `ALLOW` | $0.0$ | $+0.25$ | $0.0$ | $0.0$ | **$+0.075$** |
| **Early Attack (Ph 0)** | `BLOCK` | $+1.0$ | $0.0$ | $-0.005$ | $+1.0$ | **$+0.499$** |
| **Late Attack (Ph 3)** | `BLOCK` | $+1.0$ | $0.0$ | $-0.005$ | $+0.05$ | **$+0.357$** |
| **False Positive** | `BLOCK` | $0.0$ | $-1.2$ | $-0.005$ | $0.0$ | **$-0.361$** |
| **Missed Attack** | `ALLOW` | $-2.0$ | $0.0$ | $0.0$ | $0.0$ | **$-0.700$** |

---

## 🛡️ Anti-Degeneracy Controls

To prevent agents from learning "lazy" policies (like blocking everything or allowing everything), the environment implements:

1. **Reward Balancing**: The ratio of Miss Penalty to FP Penalty is tuned (~2.3:1) so that on a typical 80/20 traffic mix, a `block_all` policy yields a negative total reward.
2. **Pass/Fail Constraints**: Graders in [graders.py](file:///c:/Users/vettrivel/Documents/GitHub/meta_ai_hackathon/src/adaptive_firewall_env/server/graders.py) require a minimum detection rate **AND** a minimum availability rate to pass a task, regardless of the scalar reward.
