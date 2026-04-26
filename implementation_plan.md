# Codebase Analysis & Ollama Qwen3.5 Fallback Integration

## Codebase Analysis Summary

I've reviewed all **18 source files** across the entire project. Here's a comprehensive analysis:

---

## Issues Found

### 🔴 Critical Issues

#### 1. `inference.py` — `API_BASE_URL` crashes if env var is missing (Line 23)
```python
API_BASE_URL = os.environ["API_BASE_URL"]  # KeyError if missing!
API_KEY = os.environ["API_KEY"]            # Same problem
```
These use `os.environ[...]` which raises `KeyError` at import-time if the env vars aren't set. The `API_KEY` on line 25 has the same issue. During local development/testing, this crashes immediately.

> [!IMPORTANT]
> **Fix:** Change to `os.getenv()` with sensible defaults, and add the Ollama fallback here.

#### 2. `server/app.py` — `NetworkStatsResponse` model mismatch (Lines 205-207)
The endpoint `GET /stats` returns `NetworkStatsResponse(**env.get_network_stats())` but `get_network_stats()` now returns **extra fields** (`false_flag_accuracy`, `stealth_detection_rate`, `burst_ticks`, `false_flags_seen`, `stealth_attacks_seen`, `config_params`) that **are not defined** in the `NetworkStatsResponse` Pydantic model. This will cause a validation error or silently drop fields depending on Pydantic config.

#### 3. `server/firewall_environment.py` — INSPECT action bug (Lines 410-414)
```python
if inspected and session_id not in self.inspected_sessions:
    metadata["revealed"] = True
    self.inspected_sessions[session_id] = session
    self.pending_sessions[session_id] = session  # ← BUG
```
After popping from `pending_sessions` on line 400, the session is re-added to both `inspected_sessions` AND `pending_sessions`. This creates a **duplicate reference** — the session exists in both pools. When the session is later acted upon again (block after inspect), the code pops from `inspected_sessions` (line 397) but then also tries `pending_sessions.pop()` on line 441. This is not necessarily a crash, but it means:
- The session count in `state()` double-counts inspected sessions
- `_rebuild_queue` already deduplicates, so functional behavior is OK
- But the `pending_session_count` metric is inflated

---

### 🟡 Moderate Issues

#### 4. `data_loader.py` — Session TTL hardcoded vs config-driven (Lines 480-481)
```python
ttl = 2 if malicious else 3  # Hardcoded, ignoring config!
```
The `_build_session` method hardcodes TTL values, but `_spawn_sessions()` in `firewall_environment.py` **overwrites** these at lines 537-540. This is **not a bug** (the overwrite works), but it's dead code that could confuse developers.

#### 5. `server/app.py` — `StepResponse` model mismatch (Line 186)
`env.step()` returns `info.score` and `info.passed` fields, but `StepResponse.info` is typed as `Dict[str, Any]`, so this works. However `StepResponse.state` uses `StateResponse` which doesn't include `focus_observation` as a `List[float]` — this could cause validation issues in edge cases.

#### 6. Unused import in `models.py` (Line 3)
`List` from typing is imported but `Callable` is not — and `Callable` is imported in `graders.py` line 13 directly. Minor, no functional impact.

---

### 🟢 Things That Are Correct

| Component | Status | Notes |
|-----------|--------|-------|
| **Reward Engine** | ✅ Correct | Multi-objective weights sum to 1.0, anti-degenerate policy design is sound |
| **Threat Engine** | ✅ Correct | Kill chain model, stealth blending, escalation modifiers all work correctly |
| **Traffic Generator** | ✅ Correct | 22-dim features, 5 benign profiles, 5×4 malicious profiles, normalization bounds |
| **Grading System** | ✅ Correct | Deterministic seeding, weighted scoring, pass constraints |
| **Heuristic Agent** | ✅ Correct | 8-rule policy matches between `inference.py` and `heuristic_agent.py` |
| **Task Configs** | ✅ Correct | Monotonic difficulty progression across all 8 new parameters |
| **Docker Setup** | ✅ Correct | Proper non-root user, healthcheck, port exposure |
| **OpenEnv YAML** | ✅ Correct | Matches implementation: tasks, tools, action/observation spaces |
| **Test Suite** | ✅ Correct | 14 tests covering generators, rewards, threats, environment, graders |
| **Client** | ✅ Correct | Clean REST client matching all server endpoints |

---

## Proposed Changes

### 1. Ollama Qwen 3.5 Fallback in `inference.py`

The user wants the fallback to use **Ollama with Qwen 3.5** (likely `qwen3:0.6b` or `qwen2.5:3b` via Ollama cloud). I'll update `inference.py` to:

- Add Ollama as the fallback LLM provider when the primary API fails
- Use `qwen2.5:3b` (closest to "qwin3.5" — Qwen 2.5 3B is the widely-available Ollama model)
- Keep the heuristic as the final safety-net fallback

> [!IMPORTANT]
> **Clarification needed:** "qwin3.5 cloud" — I'm interpreting this as **Qwen 2.5 3B** via Ollama's local server (`http://localhost:11434/v1`). Ollama uses OpenAI-compatible API, so we can reuse the same `OpenAI` client.
> 
> If you mean a different model (e.g., `qwen3:0.6b`, `qwen2.5:7b`, or a cloud-hosted Qwen endpoint), please let me know and I'll adjust.

#### [MODIFY] [inference.py](file:///c:/Users/LE/OneDrive/Documents/GitHub/meta_ai_hackathon/inference.py)

- Add `OLLAMA_BASE_URL` and `OLLAMA_MODEL` environment variables with defaults
- Create a secondary `OpenAI` client pointing to Ollama
- In `get_action()`, on primary API failure → try Ollama → then heuristic
- Fix `API_BASE_URL` and `API_KEY` to use `os.getenv()` with defaults

---

### 2. Fix `NetworkStatsResponse` model mismatch

#### [MODIFY] [models.py](file:///c:/Users/LE/OneDrive/Documents/GitHub/meta_ai_hackathon/models.py)

- Add the missing fields to `NetworkStatsResponse`: `false_flag_accuracy`, `stealth_detection_rate`, `burst_ticks`, `false_flags_seen`, `stealth_attacks_seen`, `config_params`

---

### 3. Fix INSPECT duplicate session bug

#### [MODIFY] [firewall_environment.py](file:///c:/Users/LE/OneDrive/Documents/GitHub/meta_ai_hackathon/server/firewall_environment.py)

- Remove the line that re-adds the session to `pending_sessions` after INSPECT (line 414). The session should only be in `inspected_sessions` during the inspection phase.

---

## Open Questions

> [!IMPORTANT]
> 1. **Qwen model version**: I'm defaulting to `qwen2.5:3b` via Ollama at `http://localhost:11434/v1`. Should I use a different model name or a remote Ollama endpoint URL?
> 2. **API_KEY fallback**: Should the Ollama fallback use `"ollama"` as the API key (Ollama doesn't require one), or do you have a specific cloud-hosted Ollama endpoint that needs authentication?

---

## Verification Plan

### Automated Tests
- Run `pytest tests/` to verify no regressions
- Run `python scripts/check_accuracy.py` to validate all parameter checks pass

### Manual Verification
- Test inference with Ollama running locally to verify the fallback chain works
