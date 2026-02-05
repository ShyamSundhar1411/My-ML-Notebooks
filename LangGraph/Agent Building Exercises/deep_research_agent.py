import os,re, sys
import asyncio
from dotenv import load_dotenv
from typing import TypedDict,List,Dict,Any
from rich.live import Live
from rich.console import Console
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.markdown import Markdown
from langgraph.graph import StateGraph,START,END
from langchain_ollama import ChatOllama

from tavily import TavilyClient
import httpx
from bs4 import BeautifulSoup
from readability import Document

load_dotenv()
console = Console()
class ResearchState(TypedDict):
    user_prompt: str
    clarifications: Dict[str,str]
    needs_clarification: bool
    subqueries: List[str]
    hits: List[Dict[str,Any]]
    pages: List[Dict[str,Any]]
    report_markdown:str
    
llm = ChatOllama(model="qwen2.5:3b",temperature=0)
tavily = TavilyClient(
    api_key = os.getenv("TAVILY_API_KEY")
)
async def web_search(query:str,k:int=5)->List[Dict[str,Any]]:
    res =  tavily.search(query=query,max_results=k)
    hits = []
    for r in res.get("results",[]):
        hits.append({
            "url": r.get("url"),
            "snippet":r.get("content") or r.get("snippet"),
            "title": r.get("title"),
            "score":r.get("score",0),
        })
    return hits

async def fetch_page(url:str,timeout=15) -> Dict[str,Any]:
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.get(url)
            doc = Document(response.text)
            title = doc.short_title()
            cleaned = doc.summary(html_partial=True)
            soup = BeautifulSoup(cleaned,"html.parser")
            text = re.sub(
                r"\s+"," ",soup.get_text(" ",strip=True)
            )
            return {
                "url":str(response.url()),
                "title":title,
                "text":text[:120000]
            }
    except Exception as e:
        return {
            "url":url,
            "title":"",
            "text":f"Error fetching: {e}"
        }

def dedupe_hits(hits:List[Dict[str,Any]],limit = 25)->List[Dict[str,any]]:
    seen = set()
    unique = []
    for h in sorted(hits,key=lambda x:x.get("score",0),reverse=True):
        key = re.sub(r"#.*$","",(h.get("url") or "").strip())
        if key and key not in seen:
            seen.add(key)
            unique.append(h)
            if len(unique) >= limit:
                break
    return unique

async def node_clarify(state:ResearchState) -> ResearchState:
    prompt = f"""" You are a research agent that helps users find information online.

    Current task: {state["user_prompt"]}
    List any clarifying questions that would materially improve the research plan.
    Return JSON with keys:
        - needs: boolean
        - questions list of short questions (max 5)
    If nothing is needed, needs = false and questions= []
    """
    console.rule("[bold cyan]Clarify")
    with console.status("Thinking about what to clarify..."):
        msg = await llm.ainvoke(prompt)
    text = msg.content
    needs = "true" in text.lower()
    questions = re.findall(r"- (.+)",text) if needs else []
    if needs and questions:
        console.print(Panel("\n".join(f". {q}" for q in questions),title = "Clarifying Questions" ))
    else:
        console.print("[green]No clarifications needed")
    return {
        "needs_clarification": bool(needs and questions),
        "clarifications": {q:"" for q in questions},
        **state
        }
async def node_plan(state: ResearchState) -> ResearchState:
    clar = "\n".join(f"- {k}: {v}" for k,v in state["clarifications"].items()) or "(none)"
    prompt = f"""Create a focused research plan as 3-8 web search queries.
User prompt: {state['user_prompt']}
Clarifications:
{clar}

Rules:
- Prefer specific entities, metrics, time ranges, and synonyms.
- Cover background, current state, risks/criticisms, and data points.
Return one query per line, no numbering.
"""
    console.rule("[bold cyan]Plan")
    with console.status("Drafting sub-queries..."):
        msg = await llm.ainvoke(prompt)
    queries = [q.strip("- ").strip() for q in msg.content.splitlines() if q.strip()]
    queries = queries[:8]

    tbl = Table(title="Sub-queries", show_header=True)
    tbl.add_column("#", width=3)
    tbl.add_column("Query")
    for i,q in enumerate(queries,1):
        tbl.add_row(str(i), q)
    console.print(tbl)
    return {**state, "subqueries": queries}
async def node_search(state: ResearchState) -> ResearchState:
    console.rule("[bold cyan]Search")
    hits: List[Dict[str, Any]] = []
    with console.status("Searching the web..."):
        for q in state["subqueries"]:
            console.print(f"• [bold]Searching[/bold]: {q}")
            try:
                res = await web_search(q, k=5)
                hits.extend(res)
            except Exception as e:
                console.print(f"[red]Search failed:[/red] {e}")
    hits = dedupe_hits(hits, limit=25)

    tbl = Table(title=f"Top {len(hits)} results (deduped)", show_header=True)
    tbl.add_column("#", width=3)
    tbl.add_column("Title", overflow="fold")
    tbl.add_column("URL", overflow="fold")
    for i,h in enumerate(hits,1):
        tbl.add_row(str(i), h.get("title") or "", h.get("url") or "")
    console.print(tbl)
    return {**state, "hits": hits}
async def node_fetch(state: ResearchState) -> ResearchState:
    console.rule("[bold cyan]Fetch")
    urls = [h["url"] for h in state["hits"] if h.get("url")][:12]

    pages: List[Dict[str, Any]] = []
    for idx, u in enumerate(urls, 1):
        with console.status(f"Fetching [{idx}/{len(urls)}]: {u}"):
            p = await fetch_page(u)
        label = p.get("title") or p.get("url")
        if p.get("text","").startswith("[ERROR"):
            console.print(f"[red]✖[/red] {label}")
        else:
            console.print(f"[green]✓[/green] {label}")
        pages.append(p)

    console.print(f"Fetched {len(pages)} pages.")
    return {**state, "pages": pages}
async def node_synthesize(state: ResearchState) -> ResearchState:
    console.rule("[bold cyan]Synthesize (streaming)")
    # Build sources list & corpus
    sources = []
    for i, p in enumerate(state["pages"], start=1):
        title = p.get("title") or p.get("url")
        sources.append(f"[{i}] {title} — {p.get('url')}")
    sources_text = "\n".join(sources)

    corpus = []
    for i, p in enumerate(state["pages"], start=1):
        txt = (p.get("text") or "")[:6000]
        corpus.append(f"### Source {i}\nURL: {p.get('url')}\nTitle: {p.get('title')}\nText:\n{txt}\n")
    corpus_text = "\n".join(corpus)

    prompt = f"""Write a concise, well-structured research brief in Markdown addressing:
\"\"\"{state['user_prompt']}\"\"\".

Use sections:
- Executive Summary (5–8 bullet points)
- Key Findings
- Conflicting Views / Risks
- Data & Numbers
- Open Questions
- Next Actions
- Sources

Rules:
- Cite like [1], [2] inline after claims you derive from a source.
- Synthesize; do not just copy text.
- Prefer recent and authoritative sources.
- If sources conflict, say so.

Sources List:
{sources_text}

Corpus:
{corpus_text}
"""

    
    report_chunks: List[str] = []
    console.print(Panel("Generating report… (tokens will stream below)", style="magenta"))
    with Live("", refresh_per_second=12, console=console) as live:
        async for chunk in llm.astream(prompt):
            piece = chunk.content or ""
            report_chunks.append(piece)
            
            tail = "".join(report_chunks)[-1200:]
            live.update(Markdown(tail if tail else "."))

    report = "".join(report_chunks)
    console.rule("[bold green]Report complete")
    return {**state, "report_markdown": report}

graph = StateGraph(ResearchState)

graph.add_node("clarify", node_clarify)
graph.add_node("plan", node_plan)
graph.add_node("search", node_search)
graph.add_node("fetch",node_fetch)
graph.add_node("synthesize",node_synthesize)
graph.add_edge(START, "clarify")
def needs_more_info(state:ResearchState) ->str:
    return "plan" if not state["needs_clarification"] else "INTERRUPT"

graph.add_conditional_edges("clarify",needs_more_info,{
    "plan":"plan",
    "INTERRUPT":"clarify"
})
graph.add_edge("plan", "search")
graph.add_edge("search", "fetch")
graph.add_edge("fetch", "synthesize")
graph.add_edge("synthesize",END)
agent = graph.compile()

async def main():
    if not os.getenv("TAVILY_API_KEY"):
        console.print("[yellow]Warning:[/yellow] Missing TAVILY_API_KEY (search will fail).")

    console.print(Panel("Deep Research (streaming) — LangGraph", style="cyan"))
    user_prompt = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    if not user_prompt:
        user_prompt = console.input("[bold]What do you want to research?[/bold] ")

   
    state: ResearchState = await agent.ainvoke({
        "user_prompt": user_prompt,
        "clarifications": {},
        "needs_clarification": False,
        "subqueries": [],
        "hits": [],
        "pages": [],
        "report_markdown": ""
    })

   
    if state.get("needs_clarification"):
        console.print("[yellow]Please answer a few quick questions:[/yellow]")
        for q in state.get("clarifications", []):  
            ans = console.input(f"• {q} ")
            state["clarifications"][q] = ans
        state["needs_clarification"] = False
        state = await agent.ainvoke(state, start_at="plan")

    
    report = state["report_markdown"]
    console.print(Panel("Final Report (Markdown)", style="green"))
    console.print(Markdown(report))

    
    fname = "research_report.md"
    with open(fname, "w", encoding="utf-8") as f:
        f.write(report)
    console.print(f"[dim]Saved to ./{fname}[/dim]")

if __name__ == "__main__":
    asyncio.run(main())