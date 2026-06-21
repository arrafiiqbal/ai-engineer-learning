import requests
import streamlit as st

# ── Config ───────────────────────────────────────────────────
API_URL = "https://ai-engineer-learning-production.up.railway.app/ask"

st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="🕵️",
    layout="centered"
)

# ── Header ───────────────────────────────────────────────────
st.title("🕵️ AI Data Analyst Agent")
st.caption("Ask about sales data, AI concepts, or both — the agent decides which tools to use.")

# ── Sidebar ──────────────────────────────────────────────────
with st.sidebar:
    st.header("💡 Try asking")
    st.markdown("**📊 Business data**")
    st.caption("• What's our total revenue by category?")
    st.caption("• Who is our top customer?")
    st.markdown("**📚 AI concepts**")
    st.caption("• What is RAG?")
    st.caption("• What are embeddings?")
    st.markdown("**🔀 Both at once**")
    st.caption("• Who's our top customer and what are embeddings?")
    st.divider()
    st.caption("Powered by a LangGraph agent deployed on Railway.")

# ── Chat history ─────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "tools_used" in message and message["tools_used"]:
            with st.expander("🔧 Tools the agent used"):
                for tool in message["tools_used"]:
                    st.caption(f"• {tool}")

# ── Chat input ───────────────────────────────────────────────
if query := st.chat_input("Ask a question..."):

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🤖 The agent is thinking..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"question": query},
                    timeout=60
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "No answer returned.")
                tools_used = data.get("tools_used", [])

                st.markdown(answer)

                if tools_used:
                    with st.expander("🔧 Tools the agent used"):
                        for tool in tools_used:
                            st.caption(f"• {tool}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "tools_used": tools_used
                })

            except requests.exceptions.Timeout:
                st.error("The request timed out. The agent may be busy — try again.")
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach the agent API: {e}")
