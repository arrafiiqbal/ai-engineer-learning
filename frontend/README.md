# Frontend — AI Data Analyst Agent UI

A lightweight Streamlit chat interface for the AI Data Analyst Agent.

## Architecture

This frontend is fully decoupled from the agent logic. It makes HTTP
requests to the backend API (deployed separately on Railway) and renders
the responses — it contains no LLM, agent, or ML dependencies itself.


## Why separate?

- **Independent scaling** — UI and agent scale on their own
- **Lightweight deploys** — frontend installs only `streamlit` + `requests`
- **Realistic structure** — mirrors how production frontend/backend are split
