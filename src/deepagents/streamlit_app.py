"""
Deep Agents Chatbot — Streamlit app
====================================
A conversational chatbot built on the `deepagents` library that demonstrates
EVERY feature covered in the deepagentsdemo notebooks:

1-basicsdeepagent.ipynb   -> create_deep_agent, custom model, custom system
                             prompt, custom tools (Tavily web search),
                             built-in planning (write_todos) + virtual files
2-contextengineering.ipynb -> AGENTS.md context file, memory=, checkpointer +
                             thread_id conversation memory, Skills (/skills/)
3-backends.ipynb          -> StateBackend / FilesystemBackend / StoreBackend
4-subagents.ipynb         -> custom subagents (research-agent) + structured
                             output subagent (Pydantic response_format)

Run with:  streamlit run streamlit_app.py
"""

import os
import uuid
from pathlib import Path
from typing import Literal

import streamlit as st
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Environment (notebook 1: load API keys from .env)
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT_DIR / "src" / "deepagents"

load_dotenv(ROOT_DIR / ".env")



from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, StateBackend, StoreBackend
from deepagents.backends.utils import create_file_data
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from tavily import TavilyClient


tavily_client = TavilyClient(api_key = os.getenv("TAVILY_API_KEY"))


def internet_search(
        query: str, 
        max_results : int = 5, 
        topic: Literal["general", "news", "finanace"] = "general", 
        include_raw_content: bool = False
): 
    """Run a web search"""
    return tavily_client.search(
        query, 
        max_results=max_results, 
        include_raw_content=include_raw_content, 
        topic = topic
    )


# Structured output schema 
class ResearchFindings(BaseModel): 
    """Structured Findings from a research task"""

    summary: str = Field(description="Summary of findings")
    confidence: float = Field(description = "Confidence score from 0 to 1")
    sources: list[str] = Field(description="List of source URLs")


# Context Engineering Helpers 
def load_agents_md() -> str: 
    path = DEMO_DIR/"projects" / 'AGENTS.md'
    return path.read_text(encoding='utf-8') if path.exists() else ""

def load_skill_seed_files() -> dict: 
    """Read every file under deepagents/skills/ and convert it to in-state
        files data so the StateBackend agent can discover and read skills"""
    files = {}
    skills_root = DEMO_DIR/ "skills"
    if skills_root.exists(): 
        for f in skills_root.rglob("*.md"): 
            virtual = "/skills/" + f.relative_to(skills_root).as_posix()
            files[virtual] = create_file_data(f.read_text(encoding='utf-8'))

    return files 



# Agent Factory - assembles ALL the features based on sidebar config 
DEFAULT_SYSTEM_PROMPT = (
    "You are an expert AI assistant and researcher. You conduct thorough "
    "research using your internet_search tool when needed, plan multi-step "
    "work with write_todos, offload bulky content to files, use your skills "
    "when a query matches one, and delegate deep-dive research to your "
    "subagents. Always cite sources when research was involved."
)

SUBAGENT_DOC = """
    - **research-agent** — in-depth research with web search (context quarantine)
    - **structured-researcher** — returns `ResearchFindings` (summary, confidence, sources)
"""


def build_agent(cfg: dict): 
    """Create a deep agent wired upo according to the sidebar configuration"""
    seed_files = {}

    # Backend Selection 
    if cfg["backend"] == "StateBackend (in-state, per thread)": 
        backend = StateBackend()
        # StateBackend has no disk access -. send AGENTS.md + skills into state 
        if cfg["use_agents_md"]: 
            seed_files['/projects/AGENTS.md'] = create_file_data(load_agents_md())
        if cfg["use_skills"]: 
            seed_files.update(load_skill_seed_files())
        memory_paths = ['/projects/AGENTS.md'] if cfg['use_agents_md'] else None

    elif cfg['backend'] == "FilesystemBackend (real disk)": 
        # virtual mode = True confines the agent inside deepagents 
        backend = FilesystemBackend(root_dir=(DEMO_DIR), virtual_mode=True)
        # AGENTS.md and skills/ already exist on disk - nothing to seed 
        memory_paths = ['/projects/AGENTS.md'] if cfg["use_agents.md"] else None 

    else: # Store Backend (cross-thread memory)
        store = st.session_state.store
        backend = StoreBackend(store = store, namespace=lambda rt: ('memories',))
        # Seed durable memory inot the store once per session 
        if not st.session_state.get("store_seeded"): 
            if cfg["use_agents_md"]: 
                store.put(("memories",), "/projects/AGENTS.md", create_file_data(load_agents_md()))
            if cfg["use_skills"]: 
                for path, data in load_skill_seed_files().items(): 
                    store.put(("memories",), path, data)
            st.session_state.store_seeded = True
        memory_paths = ['/projects/AGENTS.md'] if cfg["use_agents.md"] else None

    # sub agents 
    subagents = []
    if cfg["use_subagents"]: 
        subagents.append({
            "name": "research-agent", 
            "description": "Used to research in depth questions", 
            "system_prompt": "You are a great researcher. Research thoroughly and cite your sources", 
            "tools": [internet_search]
        })
        subagents.append({
            "name": "structured-researcher", 
            "description": "Researches topics and returns structured findings"
                            "(summary, confidence score, source URLs)", 
            "system_prompt": "Research the given topic thoroughly"
                            "Return your findings", 
            "tools": [internet_search], 
            "response_format": ResearchFindings
        })


    # Aseemble the deep agent 
    kwargs = dict(
        model = cfg["model"], 
        tools = [internet_search], 
        system_prompt = cfg["system_prompt"], 
        backend = backend,
        checkpointer = st.session_state.checkpointer
    )

    if subagents: 
        kwargs["subagents"] = subagents
    if cfg["use_skills"]: 
        kwargs['skills'] = ['/skills/']
    if memory_paths: 
        kwargs["memory"] = memory_paths
    if cfg["backend"].startswith("StoreBackend"): 
        kwargs['store'] = st.session_state.store

    return create_deep_agent(**kwargs), seed_files


# Rendering helpers 
def extract_text(content) -> str: 
    """AIMessage.content may be a plain string or a list of content blocks"""
    if isinstance(content, str): 
        return content
    if isinstance(content, list): 
        parts = []
        for block in content: 
            if isinstance(block, dict) and block.get("type") == "text": 
                parts.append(block.get('text', ''))
            elif isinstance(block, str): 
                parts.append(block)
        return "\n".join(parts)
    return str(content)

def render_steps(messages): 
    """Show the agent's intermediate work: tool calls, todos, subagent tasks"""
    for msg in messages: 
        msg_type = getattr(msg, "type", "")
        if msg_type == "ai" and getattr(msg, "tool_calls", None): 
            for tc in msg.tool_calls: 
                name, args = tc["name"], tc["args"]
                if name == "write_todos": 
                    with st.expander("📋 Planning — write_todos", expanded=False):
                        for todo in args.get("todos", []): 
                            icon = {"pending": "⬜", "in_progress": "🔄",
                                    "completed": "✅"}.get(todo.get("status"), "⬜")
                            st.markdown(f"{icon} {todo.get('content', todo)}")
                elif name == "task": 
                    with st.expander(
                        f"🤖 Subagent — {args.get('subagent_type', 'task')}",
                        expanded=False,
                    ):
                        st.markdown(args.get("description", ""))
                elif name == "internet_search":
                    with st.expander(
                        f"🔎 Web search — “{args.get('query', '')}”", expanded=False
                    ):
                        st.json(args)
                elif name in ("write_file", "edit_file", "read_file", "ls",
                              "glob", "grep"):
                    label = args.get("file_path") or args.get("path") or ""
                    with st.expander(f"📁 File system — {name} {label}",
                                     expanded=False):
                        st.json(args)
                else:
                    with st.expander(f"🛠️ Tool — {name}", expanded=False):
                        st.json(args)
        elif msg_type == "tool": 
            text = extract_text(msg.content)
            if len(text) > 700:
                text = text[:700] + " …(truncated)"
            with st.expander(f"↩️ Result — {getattr(msg, 'name', 'tool')}",
                             expanded=False):
                st.code(text)


def render_files(files: dict): 
    if not files: 
        return 
    with st.expander(f"🗂️ Virtual files in state ({len(files)})", expanded=False):
        for path, data in files.items(): 
            content = data.get("content", "") if isinstance(data, dict) else str(data)
            st.markdown(f"**`{path}`**")
            st.code(content[:1500] + ("...(truncated)" if len(content)> 1500 else ""))