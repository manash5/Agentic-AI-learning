---
name: langgraph
description: LangGraph graphs, agents, state, routing, memory, persistence, streaming, and human-in-the-loop.
---

# LangGraph Skill

Use for LangGraph or LangChain graph and agent questions.

## Rules

- Design minimal state first with `TypedDict` or `MessagesState`.
- Return partial updates from nodes; do not mutate state in place.
- Use `START`, `END`, static edges, and conditional routing deliberately.
- Checkpointed apps need a configurable `thread_id`.
- Use reducers for keys written by multiple nodes.
- Prefer `create_react_agent` or `create_deep_agent` when suitable.
- For streaming, choose `updates`, `values`, or `messages` based on the UI need.
