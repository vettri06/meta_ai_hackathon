# API Reference

All endpoints are implemented in `server/app.py`.

## Core Environment Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/reset` | start a new episode (`task`, optional `seed`) |
| `POST` | `/step` | multi-session step with action map |
| `POST` | `/step_single` | single-session RL step (`action`) |
| `GET` | `/state` | current environment snapshot |
| `GET` | `/tools` | discover supported tool functions |
| `GET` | `/health` | liveness check |

## Tool Endpoints

- `POST /tool/evaluate_session`
  - body: `{ "kwargs": { "session_id": "..." } }`
- `POST /tool/take_action`
  - body: `{ "kwargs": { "session_id": "...", "action": 1 } }`
- `POST /tool/get_network_stats`
  - body: `{ "kwargs": {} }`
- `POST /tool/get_threat_intelligence`
  - body: `{ "kwargs": {} }`

## Typical Loop

1. `POST /reset`
2. `GET /state` to list candidate sessions
3. `POST /tool/evaluate_session` for selected sessions
4. `POST /step` or `POST /step_single`
5. repeat until `done=true`
