# Deep Agent Project Context

This project uses `deepagents` on LangGraph for multi-step assistants.

## Capabilities
- Plan complex work with `write_todos`.
- Use virtual file tools to offload large results and drafts.
- Delegate self-contained research to isolated subagents.
- Use project skills when a request matches one.

## Operating Rules
1. For work with more than two steps, create and update a todo list.
2. Keep chat context small: write bulky research or drafts to files and summarize them.
3. Delegate focused research with a clear, self-contained task.
4. Cite sources when web research is used.
5. Prefer the simplest correct solution and keep final answers concise.

## Project Patterns
- Models use `provider:model` strings, for example `groq:qwen/qwen3-32b`.
- Tools are Python callables, including the Tavily `internet_search` tool.
- Checkpointed conversations require a configurable `thread_id`.
- Use the configured backend for files; do not assume real-disk access.
