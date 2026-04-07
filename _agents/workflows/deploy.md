# Deploy Workflow

1. Build runtime artifact:
   - Docker image from `src/adaptive_firewall_env/server/Dockerfile`.
2. Run pre-deploy checks:
   - `pytest -q`
   - `ruff check src tests`
   - baseline evaluator output generation.
3. Publish container or code to target hosting environment.
4. Post-deploy validation:
   - `GET /health`
   - `POST /reset`
   - `POST /step_single`
5. Compare deployed baseline report with local deterministic report.
