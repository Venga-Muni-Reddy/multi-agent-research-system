"""Multi-Agent Research System: planner -> retriever -> analyst -> writer -> critic, sharing state via LangGraph."""
import os, time, json, re, asyncio
from collections import defaultdict, deque
from typing import TypedDict, List, Optional
from urllib.parse import urlparse

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END

OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
MODEL = os.environ.get("OPENROUTER_MODEL", "meta-llama/llama-3.1-8b-instruct")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS_PER_CALL", "600"))   # hard cap per LLM call
PER_IP_PER_HOUR = int(os.environ.get("PER_IP_PER_HOUR", "8"))
GLOBAL_PER_DAY = int(os.environ.get("GLOBAL_PER_DAY", "100"))

app = FastAPI(title="Multi-Agent Research System")
_hits = defaultdict(deque)
_day = {"start": time.time(), "n": 0}



class State(TypedDict, total=False):
    question: str
    prior: List[dict]
    plan: List[str]
    sources: List[dict]
    analysis: str
    report: str
    claims: List[dict]
    trace: List[str]


async def llm(system: str, user: str, max_tokens: int = None) -> str:
    if not OPENROUTER_KEY:
        raise HTTPException(503, "LLM key not configured")
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENROUTER_KEY}"},
            json={"model": MODEL, "max_tokens": max_tokens or MAX_TOKENS, "temperature": 0.3,
                  "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]},
        )
    if r.status_code != 200:
        raise HTTPException(502, f"LLM error {r.status_code}")
    return r.json()["choices"][0]["message"]["content"].strip()


def _ctx(prior):
    if not prior:
        return ""
    return "\n\nEarlier in this research thread:\n" + "\n".join(f"Q: {p['question']}\nFindings: {p['report'][:700]}" for p in prior[-2:])


async def planner(s: State) -> State:
    prior = s.get("prior") or []
    sysmsg = "You are a research planner. Return 3 short, distinct keyword-style web search queries (3 to 7 words each, no full sentences, keep the main topic terms exactly as written in the question), one per line, no numbering."
    if prior:
        sysmsg += " The question is a follow-up: use the earlier thread to make the queries specific and self-contained (resolve pronouns like 'it' or 'that')."
    out = await llm(sysmsg, s["question"] + _ctx(prior))
    qs = [q.strip("-* 0123456789.").strip().strip('"').strip() for q in out.splitlines() if q.strip()][:2]
    qs = [s["question"]] + [q for q in qs if q.lower() != s["question"].lower()]
    return {"plan": qs, "trace": s.get("trace", []) + [f"planner: {len(qs)} queries"]}


ADULT = ("porn", "xxx", "xvideos", "xnxx", "xhamster", "redtube", "youporn", "onlyfans", "sex", "nsfw", "hentai", "escort", "camgirl")
STOPW = set("what how does the are and for with that this from have has into about between which when where why who can you your their there than then them they will would should could not but also work works".split())


def _keywords(text: str):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 3 and w not in STOPW}


def _clean(hit: dict, kws: set) -> bool:
    url = (hit.get("href") or "").lower()
    host = (urlparse(url).hostname or "")
    blob = (hit.get("title", "") + " " + (hit.get("body") or "")).lower()
    if not url.startswith("http") or any(a in host or a in url.split("?")[0] for a in ADULT):
        return False
    if any(a in blob for a in ("porn", "xxx", "nsfw", "hentai")):
        return False
    return not kws or any(k in blob or k in url for k in kws)   # must mention at least one question keyword


def _search(q: str, kws: set):
    from ddgs import DDGS
    for backend in ("yahoo", "auto"):
        try:
            with DDGS(timeout=10) as d:
                hits = list(d.text(q, region="us-en", safesearch="on", max_results=6, backend=backend))
        except Exception:
            continue
        good = [h for h in hits if _clean(h, kws)][:3]
        if good:
            return good
    return []


# Heuristic source credibility by domain type. A rule-based signal, not a fact check.
HIGH = ("who.int", "nih.gov", "nature.com", "science.org", "arxiv.org", "ieee.org", "acm.org", "sciencedirect.com", "springer.com", "nasa.gov", "mit.edu", "stanford.edu")
MID_HIGH = ("wikipedia.org", "britannica.com", "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "nytimes.com", "theguardian.com", "economist.com", "ibm.com", "microsoft.com", "google.com", "aws.amazon.com", "mozilla.org", "python.org", "github.com", "docs.", "langchain.com", "openai.com", "anthropic.com", "nvidia.com", "huggingface.co")
LOW = ("reddit.com", "quora.com", "medium.com", "pinterest.", "facebook.com", "twitter.com", "x.com", "tiktok.com", "blogspot.", "substack.com", "linkedin.com", "youtube.com")


def credibility(url: str) -> dict:
    host = (urlparse(url).hostname or "").lower().removeprefix("www.")
    if host.endswith(".gov") or host.endswith(".edu") or ".ac." in host or host.endswith(".gov.in") or host.endswith(".edu.in") or any(host == d or host.endswith("." + d) for d in HIGH):
        return {"score": 90, "label": "High", "why": "Government, university or major scientific publisher"}
    if any(host == d or host.endswith("." + d) or (d.endswith(".") and host.startswith(d)) for d in MID_HIGH):
        return {"score": 75, "label": "Good", "why": "Established reference, news or vendor documentation"}
    if any(d in host for d in LOW):
        return {"score": 40, "label": "Low", "why": "User-generated or social platform"}
    if host.endswith(".org"):
        return {"score": 65, "label": "Fair", "why": "Organisation site, unverified"}
    return {"score": 55, "label": "Unrated", "why": "Unknown publisher"}


async def retriever(s: State) -> State:
    seen, src = set(), []
    results = await asyncio.gather(*[asyncio.to_thread(_search, q, _keywords(s["question"] + " " + q)) for q in s["plan"]])
    for hits in results:
        for h in hits:
            u = h.get("href")
            if u and u not in seen:
                seen.add(u)
                src.append({"title": h.get("title", ""), "url": u, "snippet": (h.get("body") or "")[:400], "credibility": credibility(u)})
    src = src[:8]
    return {"sources": src, "trace": s["trace"] + [f"retriever: {len(src)} sources"]}


def _fmt(src):
    return "\n".join(f"[{i+1}] {x['title']} - {x['snippet']}" for i, x in enumerate(src))


async def analyst(s: State) -> State:
    out = await llm("You are a research analyst. From the numbered sources, list the key facts relevant to the question as short bullets, citing source numbers like [2]. Do not invent facts. If the sources are irrelevant to the question, say so plainly in one line instead of answering from memory.",
                    f"Question: {s['question']}{_ctx(s.get('prior'))}\n\nSources:\n{_fmt(s['sources'])}")
    return {"analysis": out, "trace": s["trace"] + ["analyst: findings extracted"]}


async def writer(s: State) -> State:
    out = await llm("You are a report writer. Write a concise, well-structured report (short intro, 3-5 bullet findings, one-line conclusion) answering the question using only the findings. Keep citations like [2]. If this is a follow-up, answer the follow-up directly. If the findings say the sources were irrelevant, say the sources did not cover the question and give no claims.",
                    f"Question: {s['question']}{_ctx(s.get('prior'))}\n\nFindings:\n{s['analysis']}")
    return {"report": out, "trace": s["trace"] + ["writer: report drafted"]}


def _parse_claims(txt: str):
    m = re.search(r"\[.*\]", txt, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    out = []
    for c in arr[:6]:
        if not isinstance(c, dict) or not c.get("claim"):
            continue
        v = str(c.get("verdict", "")).lower()
        v = v if v in ("supported", "partial", "unsupported") else "partial"
        refs = [int(x) for x in (c.get("sources") or []) if str(x).isdigit()]
        out.append({"claim": str(c["claim"])[:240], "verdict": v, "sources": refs[:4], "note": str(c.get("note", ""))[:200]})
    return out or None


async def critic(s: State) -> State:
    """Fact-check agent: checks the report's key claims against the retrieved sources."""
    txt = await llm(
        "You are a strict fact-checking agent. List the 4 most important factual claims in the REPORT. For each, check it ONLY against the numbered SOURCES. "
        "Reply with a JSON array only, no prose: [{\"claim\": \"...\", \"verdict\": \"supported|partial|unsupported\", \"sources\": [1,2], \"note\": \"one short reason\"}]. "
        "Use supported only if a source snippet directly states it; partial if only loosely implied; unsupported if no source backs it.",
        f"SOURCES:\n{_fmt(s['sources'])}\n\nREPORT:\n{s['report']}", max_tokens=700)
    claims = _parse_claims(txt)
    return {"claims": claims or [], "trace": s["trace"] + [f"critic: {len(claims or [])} claims checked"]}


AGENTS = [("planner", planner), ("retriever", retriever), ("analyst", analyst), ("writer", writer), ("critic", critic)]
g = StateGraph(State)
for name, fn in AGENTS:
    g.add_node(name, fn)
g.set_entry_point("planner")
for (a, _), (b, _) in zip(AGENTS, AGENTS[1:]):
    g.add_edge(a, b)
g.add_edge("critic", END)
graph = g.compile()


class Turn(BaseModel):
    question: str = Field(max_length=300)
    report: str = Field(max_length=4000)


class Req(BaseModel):
    question: str = Field(min_length=5, max_length=300)
    thread: List[Turn] = Field(default_factory=list, max_length=6)


def _gate(request: Request):
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


def _summary(name, upd):
    if name == "planner":
        return {"queries": upd["plan"]}
    if name == "retriever":
        return {"sources": [{"title": x["title"], "url": x["url"], "credibility": x["credibility"]} for x in upd["sources"]]}
    if name == "analyst":
        return {"text": upd["analysis"]}
    if name == "writer":
        return {"text": upd["report"]}
    if name == "critic":
        return {"claims": upd["claims"]}
    return {}


@app.post("/research")
async def research(req: Req, request: Request):
    _gate(request)
    res = await graph.ainvoke({"question": req.question.strip(), "prior": [t.model_dump() for t in req.thread], "trace": []})
    return {"question": req.question, "plan": res["plan"], "sources": res["sources"], "report": res["report"], "claims": res.get("claims", []), "trace": res["trace"]}


@app.post("/research/stream")
async def research_stream(req: Req, request: Request):
    """Server-sent events: one event per agent as it starts and finishes."""
    _gate(request)

    async def gen():
        state = {"question": req.question.strip(), "prior": [t.model_dump() for t in req.thread], "trace": []}
        send = lambda o: f"data: {json.dumps(o)}\n\n"
        try:
            for name, fn in AGENTS:
                yield send({"type": "start", "agent": name})
                t0 = time.time()
                task = asyncio.create_task(fn(state))
                while not task.done():
                    await asyncio.wait({task}, timeout=4)
                    if not task.done():
                        yield ": ping\n\n"
                upd = task.result()
                state.update(upd)
                yield send({"type": "done", "agent": name, "ms": int((time.time() - t0) * 1000), "data": _summary(name, upd)})
            yield send({"type": "result", "question": req.question, "report": state["report"], "sources": state["sources"], "claims": state.get("claims", []), "plan": state["plan"]})
        except HTTPException as e:
            yield send({"type": "error", "message": e.detail})
        except Exception as e:
            yield send({"type": "error", "message": f"{type(e).__name__}: research failed"})

    return StreamingResponse(gen(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/health")
async def health():
    return {"ok": True}


app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")


@app.get("/")
async def index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "static", "index.html"))
