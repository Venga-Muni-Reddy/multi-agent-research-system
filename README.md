# Multi-Agent Research System

FastAPI + LangGraph. Four agents (planner, retriever, analyst, report writer) share state through a LangGraph graph.

- `POST /research {"question": "..."}` returns a cited report, the plan, sources and an agent trace
- Web UI at `/`
- Config via env: `OPENROUTER_API_KEY`, optional `OPENROUTER_MODEL`, `MAX_TOKENS_PER_CALL`, `PER_IP_PER_HOUR`, `GLOBAL_PER_DAY`

## Vendor decision brief

Choosing between two tools usually means reading long documentation pages and trying to remember what each one said about your requirements. This feature does that reading for you without letting the AI invent facts.

How it works: you paste documentation text for two options and list 2 to 6 requirements. Two evidence agents (LangGraph) extract what each document says about each requirement. Every finding must include a quote, and the quote is checked against the text you pasted. If it is not found, the finding is discarded and shown as Unknown, with a question to ask the vendor. The app never picks a winner.

Endpoint: `POST /decision` (goal, criteria, options with name, source, text). Model: OpenRouter `openrouter/free` using the server-side `OPENROUTER_API_KEY`. Quotes are verified by code, but whether a quote answers the requirement is the model's judgment, so read the quotes. Do not paste confidential text, it is sent to OpenRouter.
