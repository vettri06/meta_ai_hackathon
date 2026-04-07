# Setup Workflow

1. Create virtual environment:
   - `py -m venv .venv`
2. Install dependencies:
   - `.venv\Scripts\python -m pip install -U pip`
   - `.venv\Scripts\python -m pip install pytest ruff requests numpy fastapi pydantic uvicorn`
3. Validate code quality:
   - `.venv\Scripts\python -m pytest -q`
   - `.venv\Scripts\python -m ruff check src tests`
4. Start service:
   - `uvicorn adaptive_firewall_env.server.app:app --port 8000`
5. Run baseline evaluator for smoke confirmation.
