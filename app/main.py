"""Multi-Agent Research System: planner -> retriever -> analyst -> writer, sharing state via LangGraph."""
import os, time, json
from collections import defaultdict, deque
from typing import TypedDict, List

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS_PER_CALL", "600"))   # hard cap per LLM call
PER_IP_PER_HOUR = int(os.environ.get("PER_IP_PER_HOUR", "5"))
GLOBAL_PER_DAY = int(os.environ.get("GLOBAL_PER_DAY", "60"))

app = FastAPI(title="Multi-Agent Research System")
_hits = defaultdict(deque)
_day = {"start": time.time(), "n": 0}


class State(TypedDict, total=False):
    question: str
    plan: List[str]
    sources: List[dict]
    analysis: str
    report: str
    trace: List[str]


async def llm(system: str, user: str) -> str:
    if not OPENROUTER_KEY:
        raise HTTPException(503, "LLM key not configured")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={"model": MODEL, "max_tokens": MAX_TOKENS, "temperature": 0.3,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
    if r.status_code != 200:
        raise HTTPException(502, f"LLM error {r.status_code}")
    return r.json()["choices"][0]["message"]["content"].strip()


async def planner(s: State) -> State:
    out = await llm("You are a research planner. Return 3 short, distinct web search queries for the question, one per line, no numbering.", s["question"])
    qs = [q.strip("-* 0123456789.").strip().strip('"').strip() for q in out.splitlines() if q.strip()][:3] or [s["question"]]
    return {"plan": qs, "trace": s.get("trace", []) + [f"planner: {len(qs)} queries"]}


def _search(q: str):
    from ddgs import DDGS
    try:
        with DDGS() as d:
            return list(d.text(q, region="us-en", max_results=3))
    except Exception:
        return []


async def retriever(s: State) -> State:
    import asyncio
    seen, src = set(), []
    for q in s["plan"]:
        for h in await asyncio.to_thread(_search, q):
            u = h.get("href")
            if u and u not in seen:
                seen.add(u)
                src.append({"title": h.get("title", ""), "url": u, "snippet": (h.get("body") or "")[:400]})
    return {"sources": src[:8], "trace": s["trace"] + [f"retriever: {len(src[:8])} sources"]}


def _fmt(src):
    return "\n".join(f"[{i+1}] {x['title']} - {x['snippet']}" for i, x in enumerate(src))


async def analyst(s: State) -> State:
    out = await llm("You are a research analyst. From the numbered sources, list the key facts relevant to the question as short bullets, citing source numbers like [2]. Do not invent facts.",
                    f"Question: {s['question']}\n\nSources:\n{_fmt(s['sources'])}")
    return {"analysis": out, "trace": s["trace"] + ["analyst: findings extracted"]}


async def writer(s: State) -> State:
    out = await llm("You are a report writer. Write a concise, well-structured report (short intro, 3-5 bullet findings, one-line conclusion) answering the question using only the findings. Keep citations like [2].",
                    f"Question: {s['question']}\n\nFindings:\n{s['analysis']}")
    return {"report": out, "trace": s["trace"] + ["writer: report drafted"]}


g = StateGraph(State)
for name, fn in [("planner", planner), ("retriever", retriever), ("analyst", analyst), ("writer", writer)]:
    g.add_node(name, fn)
g.set_entry_point("planner")
g.add_edge("planner", "retriever")
g.add_edge("retriever", "analyst")
g.add_edge("analyst", "writer")
g.add_edge("writer", END)
graph = g.compile()


class Req(BaseModel):
    question: str = Field(min_length=5, max_length=300)


@app.post("/research")
async def research(req: Req, request: Request):
    ip = (request.headers.get("x-forwarded-for") or request.client.host).split(",")[0].strip()
    now = time.time()
    if now - _day["start"] > 86400:
        _day.update(start=now, n=0)
    if _day["n"] >= GLOBAL_PER_DAY:
        raise HTTPException(429, "Daily demo limit reached. Please try again tomorrow.")
    q = _hits[ip]
    while q and now - q[0] > 3600:
        q.popleft()
    if len(q) >= PER_IP_PER_HOUR:
        raise HTTPException(429, f"Limit: {PER_IP_PER_HOUR} research runs per hour per visitor.")
    q.append(now); _day["n"] += 1
    res = await graph.ainvoke({"question": req.question.strip(), "trace": []})
    return {"question": req.question, "plan": res["plan"], "sources": res["sources"], "report": res["report"], "trace": res["trace"]}


@app.get("/health")
async def health():
    return {"ok": True}


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
