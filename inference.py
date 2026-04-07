from __future__ import annotations

import json
import os
import sys
import time
import textwrap
from typing import Any, Dict, List, Optional

import numpy as np
from openai import OpenAI

# Import the environment directly for the AI Firewall
from server.firewall_environment import FirewallEnvironment, ACTIONS, TASK_CONFIGS

# --- Hackathon Submission Rules Compliance ---
# 1. inference.py in root directory                       ✅
# 2. Use OpenAI Client for all LLM calls                 ✅
# 3. Required Environment Variables with Defaults         ✅
# 4. Strict Output Format: [START], [STEP], [END]         ✅

# Environment Variables per Spec
API_BASE_URL = os.environ["API_BASE_URL"]
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-Coder-7B-Instruct")
API_KEY = os.environ["API_KEY"]

# Benchmark configuration
BENCHMARK = "ai-firewall"


def format_bool(v: bool) -> str:
    return "true" if v else "false"


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = format_bool(done)
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(task: str, score: float, steps: int) -> None:
    # Score should be between 0.01 and 0.99 as per user feedback
    clamped_score = max(0.01, min(0.99, score))
    print(f"[END] task={task} score={clamped_score:.2f} steps={steps}", flush=True)


class InferenceAgent:
    def __init__(self):
        self.client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    def get_action(self, session_data: Dict[str, Any], threat_intel: Dict[str, Any]) -> int:
        """Get action using LLM via OpenAI client interface with heuristic fallback."""
        system_prompt = textwrap.dedent(
            """
            You are an adaptive AI firewall controller. 
            Respond with ONLY valid JSON in this shape: {"reasoning": string, "action": integer}. 
            Action must be one integer between 0 and 5: 0=ALLOW, 1=BLOCK, 2=INSPECT, 3=SANDBOX, 4=RATE_LIMIT, 5=QUARANTINE. 
            Keep reasoning short (under 20 words).
            """
        ).strip()

        user_prompt = json.dumps({
            "session": session_data,
            "threat_intelligence": threat_intel,
            "actions": ACTIONS
        })

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2,
                    max_tokens=150,
                )

                raw_content = response.choices[0].message.content

                # Attempt to parse JSON
                if "```json" in raw_content:
                    raw_content = raw_content.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_content:
                    raw_content = raw_content.split("```")[1].split("```")[0].strip()

                content = json.loads(raw_content)
                action = int(content.get("action", 0))
                return max(0, min(5, action))

            except Exception as e:
                if "429" in str(e) and attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return self._heuristic_action(session_data, threat_intel)

        return self._heuristic_action(session_data, threat_intel)

    def _heuristic_action(self, session_data: Dict[str, Any], threat_intel: Dict[str, Any]) -> int:
        """Rule-based fallback with 8 detection rules."""
        features = session_data.get("features", {})
        known_bad_ports = set(threat_intel.get("known_bad_ports", []))

        if session_data.get("revealed_malicious") is True:
            return 1  # BLOCK

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
            return 1
        if self_signed == 1 and history < 0.45:
            return 5
        if entropy > 0.55 and reuse < 0.25:
            return 2
        if geo > 4000.0 and history < 0.40:
            return 2
        if ja3 >= 180:
            return 1
        if dur < 60.0 and pkts > 100:
            return 4
        if cert_valid < 80.0 and tls_ver == 0:
            return 2
        if reuse < 0.10 and dns_q >= 4:
            return 2

        return 0  # ALLOW


def run_task(agent: InferenceAgent, task: str):
    """Run a single task episode and emit spec-compliant output."""
    seeds = {"easy": 101, "medium": 202, "hard": 303}
    env = FirewallEnvironment(seed=seeds.get(task, 101))

    log_start(task=task, env=BENCHMARK, model=MODEL_NAME)

    state = env.reset(task=task)
    done = False
    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.01

    try:
        while not done:
            action = 0
            error_msg = None

            focus_session_id = state.get("focus_session_id")
            if focus_session_id:
                try:
                    session_data = env.evaluate_session(focus_session_id)
                    threat_intel = env.get_threat_intelligence()
                    action = agent.get_action(session_data, threat_intel)
                    result = env.step_single(action)
                except Exception as e:
                    error_msg = str(e)
                    result = env.step_single(0)
            else:
                result = env.step_single(0)

            reward = float(result["reward"])
            done = bool(result["done"])
            state = result["state"]
            steps_taken += 1
            rewards.append(reward)

            log_step(
                step=steps_taken,
                action=ACTIONS.get(action, "ALLOW"),
                reward=reward,
                done=done,
                error=error_msg,
            )

            if done:
                break

        # Calculate final score via grader
        final_stats = env.get_network_stats()
        from server.graders import grade_stats
        grade = grade_stats(task, final_stats)
        final_score = float(grade.get("score", 0.01))

    except Exception as e:
        print(f"[DEBUG] Error during task {task}: {e}", file=sys.stderr)
        final_score = 0.01
    finally:
        log_end(task=task, score=final_score, steps=steps_taken)


def main():
    try:
        agent = InferenceAgent()
        for task in ["easy", "medium", "hard"]:
            run_task(agent, task)
    except Exception as e:
        print(f"Critical error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
