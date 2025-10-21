# Repository Guidelines

## Project Structure & Module Organization
- `mcpuniverse/` holds the production code, with key packages for agents (`agent/`), orchestration (`workflows/`), protocol adapters (`mcp/`), model clients (`llm/`), and the web stack (`dashboard/`, `app/`).
- `tests/` contains the pytest suites; mirror new modules here with matching `test_<module>.py` files to keep coverage aligned.
- `assets/`, `docs/`, and `docker/` provide static resources, additional guides, and container recipes; database migrations live in `mcpuniverse/app/db/migration/`.
- Reference shared configs such as `pyproject.toml`, `pytest.ini`, and the root `Makefile` before introducing new tooling.

## Build, Test, and Development Commands
- `pip install -r requirements.txt && pip install -r dev-requirements.txt` prepares runtime and contributor dependencies.
- `make test` (or `PYTHONPATH=. pytest tests/`) runs the full Python test suite; add `-k "<pattern>"` for focused runs.
- `make dashboard` starts the FastAPI/Gradio stack via `uvicorn mcpuniverse.dashboard.app:app`.
- `make postgres` / `make redis` bootstrap local services; pair with `make createdb` and `make new_migration name=<slug>` for database work.

## Coding Style & Naming Conventions
- Target Python 3.10+ with 4-space indentation, f-strings, and explicit typing; CI enforces `pylint`, so run `pre-commit install` and execute hooks locally.
- Follow snake_case for modules and functions, PascalCase for classes, and keep configuration files lowercase (e.g., `benchmark/configs/*.yaml`).
- Align docstrings and comments with prevalent Google-style summaries, keeping explanations concise and actionable.

## Testing Guidelines
- Use `pytest` plus the bundled plugins (`pytest-asyncio`, `pytest-postgresql`) for asynchronous flows and database fixtures.
- Co-locate fixtures under `tests/conftest.py` or module-specific `conftest.py` files to avoid cross-suite coupling.
- When introducing features, add regression tests in `tests/` and consider smoke scenarios under `mcpuniverse/benchmark/` if they exercise full agent workflows.
- Capture failing reproductions with parametrized cases and verify external service dependencies via the Docker targets above.

## Commit & Pull Request Guidelines
- Keep commits small and descriptive, mirroring the existing log style (`add some init pipedream set up`); prefer present-tense summaries with relevant components up front.
- Reference GitHub issues in commit bodies when applicable and ensure every PR states scope, test evidence, and any service toggles touched.
- Run `make test` (and service containers if required) before opening a PR; attach logs or screenshots for dashboard updates and benchmark results.
- Confirm the Salesforce CLA is signed and link to security implications per `SECURITY.md` when proposing MCP server integrations.

## Environment & Configuration Tips
- Start from `.env.example` for local credentials and extend cautiously; never commit secrets or API keys.
- Review `SECURITY.md` and `AI_ETHICS.md` when integrating third-party MCP servers or LLM providers, documenting threat mitigations in PR notes.
- Prefer `docker/` compose recipes for reproducible demos, and document deviations in `docs/` to keep peer agents aligned.
