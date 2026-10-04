# Multi-Agent Research System

FastAPI + LangGraph. Four agents (planner, retriever, analyst, report writer) share state through a LangGraph graph.

- `POST /research {"question": "..."}` returns a cited report, the plan, sources and an agent trace
- Web UI at `/`
- Config via env: `OPENROUTER_API_KEY`, optional `OPENROUTER_MODEL`, `MAX_TOKENS_PER_CALL`, `PER_IP_PER_HOUR`, `GLOBAL_PER_DAY`
