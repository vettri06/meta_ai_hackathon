# Deployment

## Local Runtime

```bash
uvicorn adaptive_firewall_env.server.app:app --host 0.0.0.0 --port 8000
```

## Container Runtime

```bash
docker build -f src/adaptive_firewall_env/server/Dockerfile -t adaptive-firewall-env .
docker run --rm -p 8000:8000 adaptive-firewall-env
```

## OpenEnv Metadata

- Manifest path: `src/adaptive_firewall_env/openenv.yaml`
- Runtime type: FastAPI app (`server.app:app`)
- Default port: `8000`

## Smoke Checks

- `GET /health` returns `{ "status": "ok" }`
- `POST /reset` returns episode state
- `POST /step_single` returns next observation and reward
