"""FastAPI server exposing the Adaptive AI Firewall environment.

Endpoints:
  POST /reset     — Start a new episode
  POST /step      — Multi-session step (batch actions)
  POST /step_single — Single-session step (Gymnasium-compatible)
  GET  /state     — Current environment state
  GET  /tools     — List available tool names
  POST /tool/{name} — Call a specific tool
  GET  /health    — Health check
  GET  /stats     — Current episode statistics
"""
from __future__ import annotations

import os
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from dotenv import load_dotenv

from server.firewall_environment import FirewallEnvironment, ACTIONS
from models import (
    HealthResponse,
    NetworkStatsResponse,
    ResetRequest,
    StateResponse,
    StepRequest,
    StepResponse,
    StepSingleRequest,
    StepSingleResponse,
    ToolRequest,
    ToolsListResponse,
)

load_dotenv()


def _clean_env_value(value: str) -> str:
    return value.strip().strip("`").strip().strip("'").strip('"').strip()


def _resolve_api_key(value: str | None) -> str:
    return _clean_env_value(value or os.getenv("HF_TOKEN") or "")


def _resolve_model(value: str | None) -> str:
    return _clean_env_value(value or os.getenv("MODEL_NAME") or "")


def _resolve_base_url(value: str | None) -> str:
    return _clean_env_value(
        value
        or os.getenv("API_BASE_URL")
        or ""
    )

PLAYGROUND_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width,initial-scale=1"/>
  <title>Adaptive Firewall Playground</title>
  <style>
    body{font-family:Arial,sans-serif;background:#0b1220;color:#e5e7eb;margin:0;padding:24px}
    .card{max-width:980px;margin:0 auto;background:#111827;border:1px solid #1f2937;border-radius:12px;padding:18px}
    h1{margin-top:0;font-size:22px}
    label{display:block;font-size:12px;margin:10px 0 4px}
    input,textarea,button{width:100%;box-sizing:border-box;border-radius:8px;border:1px solid #374151;background:#0f172a;color:#e5e7eb;padding:10px}
    textarea{min-height:120px;resize:vertical}
    button{background:#2563eb;border:none;cursor:pointer;font-weight:600;margin-top:12px}
    button:disabled{opacity:.6;cursor:not-allowed}
    pre{white-space:pre-wrap;background:#0f172a;border:1px solid #374151;border-radius:8px;padding:12px;min-height:120px;overflow:auto}
    .grid{display:grid;grid-template-columns:1fr 1fr;gap:10px}
    .row{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
    .muted{font-size:12px;color:#93c5fd}
    .ok{color:#86efac}
    .bad{color:#fca5a5}
    .btn-step{background:#22c55e}
    .btn-reset{background:#64748b}
    .btn-state{background:#64748b}
  </style>
</head>
<body>
  <div class="card">
    <h1>Playground</h1>
    <p class="muted">Click Reset to start a new episode.</p>
    
    <label>Message / Action ID</label>
    <input id="action_input" type="number" value="0" min="0" max="5" placeholder="Enter action index (0-5)..." />
    
    <div class="row">
      <button id="btn_step" class="btn-step">Step</button>
      <button id="btn_reset" class="btn-reset">Reset</button>
      <button id="btn_state" class="btn-state">Get state</button>
    </div>
    
    <div id="status" class="muted" style="margin-top:10px">Ready</div>
    
    <label>Raw JSON response</label>
    <pre id="output">{}</pre>
  </div>
  
  <script>
    const output = document.getElementById("output");
    const status = document.getElementById("status");
    const actionInput = document.getElementById("action_input");

    async function call(path, method='GET', body=null) {
      status.textContent = "Calling " + path + "...";
      try {
        const options = {
          method: method,
          headers: {"Content-Type":"application/json"}
        };
        if (body) options.body = JSON.stringify(body);
        
        const res = await fetch(path, options);
        const data = await res.json();
        output.textContent = JSON.stringify(data, null, 2);
        status.textContent = "Success";
        return data;
      } catch (err) {
        status.textContent = "Error: " + err;
        output.textContent = String(err);
      }
    }

    document.getElementById("btn_step").onclick = () => {
      const action = parseInt(actionInput.value);
      call("/step_single", "POST", {action: action});
    };

    document.getElementById("btn_reset").onclick = () => {
      call("/reset", "POST", {task: "easy"});
    };

    document.getElementById("btn_state").onclick = () => {
      call("/state", "GET");
    };
  </script>
</body>
</html>"""


env = FirewallEnvironment(seed=42)
app = FastAPI(
    title="Adaptive AI Firewall OpenEnv",
    version="0.2.0",
    description="RL environment for adaptive firewall decision making on encrypted traffic.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    """Redirect root to the playground UI."""
    return HTMLResponse(content=PLAYGROUND_HTML)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version="0.2.0")


@app.post("/reset", response_model=StateResponse)
def reset(request: ResetRequest = ResetRequest()) -> StateResponse:
    try:
        state = env.reset(task=request.task, seed=request.seed)
        return StateResponse(**state)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/step", response_model=StepResponse)
def step(request: StepRequest = StepRequest()) -> StepResponse:
    result = env.step(action_map=request.actions)
    return StepResponse(**result)


@app.post("/step_single", response_model=StepSingleResponse)
def step_single(request: StepSingleRequest = None) -> StepSingleResponse:
    if request is None:
        raise HTTPException(status_code=422, detail="Body is required for /step_single")
    try:
        result = env.step_single(action=request.action)
        return StepSingleResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/state", response_model=StateResponse)
def state() -> StateResponse:
    return StateResponse(**env.state())


@app.get("/stats", response_model=NetworkStatsResponse)
def stats() -> NetworkStatsResponse:
    return NetworkStatsResponse(**env.get_network_stats())


@app.get("/tools", response_model=ToolsListResponse)
def list_tools() -> ToolsListResponse:
    return ToolsListResponse(tools=env.list_tools())


@app.get("/web", response_class=HTMLResponse)
def web_interface() -> HTMLResponse:
    return HTMLResponse(content=PLAYGROUND_HTML)


@app.get("/schema")
def schema() -> Any:
    return {
        "observation_space": {
            "type": "Box",
            "shape": [22],
            "low": 0.0,
            "high": 1.0,
        },
        "action_space": {
            "type": "Discrete",
            "n": 6,
            "actions": ACTIONS,
        },
    }


@app.post("/tool/{name}")
def call_tool(name: str, request: ToolRequest) -> Any:
    try:
        if name == "evaluate_session":
            return env.evaluate_session(request.kwargs["session_id"])
        if name == "take_action":
            reward, record = env.take_action(
                session_id=request.kwargs["session_id"],
                action=int(request.kwargs["action"]),
            )
            return {"reward": reward, "record": record}
        if name == "get_network_stats":
            return env.get_network_stats()
        if name == "get_threat_intelligence":
            return env.get_threat_intelligence()
        raise HTTPException(status_code=404, detail=f"unknown tool: {name}")
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"missing key: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def main() -> None:
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
