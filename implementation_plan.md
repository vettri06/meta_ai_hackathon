# Adaptive AI Firewall — OpenEnv RL Challenge Compliance

## Background

Your current codebase has a solid firewall RL environment, grader, and inference agent. However, several critical areas need changes to pass the hackathon's automated validation. I've analyzed the reference repos (reasoning_gym_env, calendar_env) and the submission guidelines in detail.

## User Review Required

> [!IMPORTANT]
> **Ollama vs HuggingFace Router**: The hackathon guidelines mandate using the **OpenAI Client** with `API_BASE_URL` (default pointing to HuggingFace router) and `HF_TOKEN`. You mentioned wanting to use Ollama — but **the evaluation system will inject its own `API_BASE_URL` and `MODEL_NAME`** pointing to their hosted models. Your code must use the OpenAI client talking to whatever `API_BASE_URL` is provided. Ollama won't work during evaluation because the Docker container runs on HF Spaces with 2 vCPU / 8 GB RAM — no room to run a local LLM. Your current setup (HF router + OpenAI client) is **already correct**. I'll keep it as-is.

> [!WARNING]
> **Your `.env` file contains a real `HF_TOKEN`**. This is committed to git. You should rotate this token after we're done and add `.env` to `.gitignore`.

---

## Proposed Changes

### 1. `inference.py` — Complete Rewrite (Critical)

#### [MODIFY] [inference.py](file:///c:/Users/vettrivel/Documents/GitHub/meta_ai_hackathon/inference.py)

**Current problems:**
- ❌ No `[START]` / `[STEP]` / `[END]` output lines (the #1 compliance requirement)
- ❌ `API_BASE_URL` and `MODEL_NAME` have no default values (will fail validation)
- ❌ `HF_TOKEN` is not validated as mandatory (should raise on missing)
- ❌ Uses `argparse` — evaluation just runs `python inference.py`
- ❌ Output is JSON, not the required line format

**Changes:**
- Add default values: `API_BASE_URL="https://router.huggingface.co/v1"`, `MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct"`
- Raise `ValueError` if `HF_TOKEN` is missing
- Print `[START]` line before episode begins
- Print `[STEP]` line immediately after each `env.step()` return
- Print `[END]` line after episode ends (even on exception, using try/finally)
- Format rewards to 2 decimal places, booleans as lowercase `true`/`false`
- Remove argparse; hardcode task or pick from env var
- Keep the LLM-based agent logic (get_action) but fix it to work with defaults
- Run all 3 tasks sequentially (easy, medium, hard) or pick the best one

---

### 2. Server — Align with OpenEnv `create_app` Pattern

#### [MODIFY] [app.py](file:///c:/Users/vettrivel/Documents/GitHub/meta_ai_hackathon/src/adaptive_firewall_env/server/app.py) (primary server)

**Current problems:**
- ❌ Hand-rolled FastAPI endpoints — reference repos use `openenv.core.env_server.http_server.create_app()`
- ❌ The import chain in `server/app.py` (root) references a non-existent module path

**Changes:**
- **Keep the current hand-rolled server** since `create_app` requires `openenv.core.env_server.interfaces.Environment` base class and the firewall env doesn't extend it. Refactoring to use `create_app` would require significant env restructuring.
- Instead, fix the root `server/app.py` to correctly import from the right location
- Add `/web` endpoint for HF Spaces web interface compatibility
- Add `/schema` endpoint returning action/observation schemas

#### [MODIFY] [app.py](file:///c:/Users/vettrivel/Documents/GitHub/meta_ai_hackathon/server/app.py) (root server entry)

- Fix the broken import `from adaptive_firewall_env.server.app import app`
- Make it correctly reference the actual app

---

### 3. Dockerfile — Production Ready for HF Spaces

#### [MODIFY] [Dockerfile](file:///c:/Users/vettrivel/Documents/GitHub/meta_ai_hackathon/Dockerfile)

**Current problems:**
- ❌ Doesn't copy `inference.py` into the container
- ❌ Doesn't copy `env/`, `grader/`, `utils/` directories
- ❌ Heavy dependencies (torch, stable-baselines3) blow through 8 GB RAM

**Changes:**
- Copy ALL required source directories (`env/`, `grader/`, `utils/`, `inference.py`, `models/`)
- Set `PYTHONPATH` correctly
- Optimize requirements for smaller image size  
- Keep `CMD` as uvicorn for the server (HF Spaces), but ensure `inference.py` can also run independently

---

### 4. Requirements — Trim for 8 GB RAM Constraint

#### [MODIFY] [requirements.txt](file:///c:/Users/vettrivel/Documents/GitHub/meta_ai_hackathon/requirements.txt)

**Changes:**
- Remove `torch` (huge, not needed for inference — agent uses OpenAI API)
- Remove `stable-baselines3` (training framework, not needed at inference)
- Remove `shimmy` (adapter for SB3)
- Remove `gymnasium` (not needed if using custom env directly)
- Keep: `fastapi`, `uvicorn`, `numpy`, `pydantic`, `requests`, `openai`, `python-dotenv`

---

### 5. `.env.example` — Fix Defaults

#### [MODIFY] [.env.example](file:///c:/Users/vettrivel/Documents/GitHub/meta_ai_hackathon/.env.example)

- Document that `HF_TOKEN` is **mandatory**
- Show default values for `API_BASE_URL` and `MODEL_NAME`

---

### 6. Fix Import Chain in `utils/threat_engine.py`

#### [MODIFY] [threat_engine.py](file:///c:/Users/vettrivel/Documents/GitHub/meta_ai_hackathon/utils/threat_engine.py)

**Current problem:**
- Line 17: `from adaptive_firewall_env.server.traffic_generator import TrafficGenerator` — wrong import path
- Should be `from utils.data_loader import TrafficGenerator`

---

### 7. `.gitignore` — Protect Secrets

#### [MODIFY] [.gitignore](file:///c:/Users/vettrivel/Documents/GitHub/meta_ai_hackathon/.gitignore)

- Ensure `.env` is listed (prevent token leaks)

---

## Architecture Summary After Changes

```
meta_ai_hackathon/
├── inference.py          ← MAIN ENTRY POINT (hackathon requirement)
├── Dockerfile            ← HF Spaces deployment  
├── requirements.txt      ← Trimmed dependencies
├── openenv.yaml          ← Environment manifest
├── .env.example          ← Template with docs
├── env/
│   ├── firewall_env.py   ← Core RL environment
│   └── models.py         ← Pydantic request/response models
├── grader/
│   └── firewall_grader.py ← Scoring logic
├── utils/
│   ├── data_loader.py    ← Traffic generation
│   ├── reward_engine.py  ← Multi-objective rewards
│   └── threat_engine.py  ← Attack orchestration (import fixed)
├── server/
│   └── app.py            ← FastAPI server for HF Spaces
└── src/adaptive_firewall_env/server/
    └── app.py            ← Full server with LLM playground
```

## Open Questions

> [!IMPORTANT]  
> **Which tasks to run?** The hackathon evaluator likely runs `python inference.py` without arguments. Should we:
> - (A) Run all 3 tasks (easy, medium, hard) sequentially and output [START]/[STEP]/[END] for each?
> - (B) Run only `easy` by default?
> - I recommend **(A)** — running all 3 tasks to maximize score visibility. Each gets its own `[START]`/`[END]` block.

> [!IMPORTANT]
> **Max steps per task:** Easy=200, Medium=500, Hard=1000 steps. With LLM calls at each step, this could be slow with rate limits. Should I add a timeout or fallback more aggressively to heuristics?

## Verification Plan

### Automated Tests
1. Run `python inference.py` and verify stdout matches the exact format:
   ```
   [START] task=easy env=ai-firewall model=meta-llama/Llama-3.1-8B-Instruct
   [STEP] step=1 action=ALLOW reward=0.00 done=false error=null
   ...
   [END] success=true steps=200 rewards=0.00,0.00,...
   ```
2. Run `docker build -t ai-firewall .` and verify it builds under 8 GB
3. Run the container and hit `/health` endpoint
4. Verify all env vars work with defaults when `HF_TOKEN` is set

### Manual Verification
- Deploy to HF Spaces and confirm the space reaches "Running" state
- Verify the web interface loads at the space URL
