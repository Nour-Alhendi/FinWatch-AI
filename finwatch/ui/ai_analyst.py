"""
AI Analyst — Screen 3
Chat interface wired to the Groq-SDK FinWatchAgent.
"""

from __future__ import annotations
import streamlit as st

from ui.theme import CHART_AXIS


_SUGGESTED = [
    "What are the biggest risks in the portfolio right now?",
    "Which sector has the highest average drawdown probability?",
    "Explain why NVDA is flagged as anomalous.",
    "Compare VaR 95% for AAPL vs MSFT.",
    "Show me the market context for JPM.",
]


def _build_agent():
    import sys
    from pathlib import Path

    _root = Path(__file__).resolve().parents[3]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    from src.agent.agent import build_agent  # noqa: PLC0415
    return build_agent()


def _run_agent(agent, prompt: str) -> tuple[str, list[str]]:
    """Run the FinWatchAgent synchronously; return (answer, tool_names)."""
    try:
        return agent.run(prompt)
    except Exception as exc:
        return f"**Error:** {exc}", []


_AI_CSS = """
<style>
/* ── Chat input ── */
[data-testid="stChatInput"] textarea{
    background:#161b22!important;
    border:0.5px solid rgba(27,200,160,0.25)!important;
    color:#c9d1d9!important;
    font-family:'IBM Plex Mono',monospace!important;
    font-size:14px!important;
    border-radius:8px!important;
}
[data-testid="stChatInput"] textarea:focus{
    border-color:rgba(27,200,160,0.5)!important;
    box-shadow:0 0 0 1px rgba(27,200,160,0.15)!important;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"]{
    background:#161b22!important;
    border:0.5px solid rgba(255,255,255,0.14)!important;
    border-radius:8px!important;
    margin-bottom:8px!important;
}

/* ── Suggested prompt chips ── */
.chip-row{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:16px}
.chip-btn button{
    background:#161b22!important;
    border:0.5px solid rgba(255,255,255,0.18)!important;
    color:#7a8da3!important;
    font-family:'IBM Plex Mono',monospace!important;
    font-size:12px!important;
    border-radius:16px!important;
    padding:5px 14px!important;
    min-height:30px!important;
    transition:border-color 0.12s,color 0.12s!important;
}
.chip-btn button:hover{
    border-color:rgba(27,200,160,0.4)!important;
    color:#1bc8a0!important;
    background:#161b22!important;
}

/* ── Tool call chips ── */
.tool-chips{margin-bottom:8px}
.tool-chip{
    display:inline-block;
    font-size:11px;letter-spacing:0.3px;
    font-family:'IBM Plex Mono',monospace;
    color:#4a6580;
    background:rgba(22,27,34,0.8);
    border:0.5px solid rgba(255,255,255,0.15);
    border-radius:10px;
    padding:2px 10px;
}

/* ── Status dot ── */
.ai-dot{
    width:8px;height:8px;border-radius:50%;
    background:#1bc8a0;
    box-shadow:0 0 10px rgba(27,200,160,0.4);
    animation:pulse-ai 2.5s ease-in-out infinite;
    display:inline-block;
}
@keyframes pulse-ai{
    0%,100%{opacity:0.5;box-shadow:0 0 5px rgba(27,200,160,0.25)}
    50%{opacity:1;box-shadow:0 0 16px rgba(27,200,160,0.6)}
}

/* ── Disclaimer ── */
.ai-disclaimer{
    font-size:12px;letter-spacing:0.5px;color:#3a4d5e;
    font-family:'IBM Plex Mono',monospace;text-align:center;
    margin-top:14px;line-height:1.5;
}
</style>
"""
_AI_CSS = (
    _AI_CSS
    .replace("#7a8da3", "#9aadbd")
    .replace("#4a6580", CHART_AXIS)
    .replace("#3a4d5e", CHART_AXIS)
)


def render_ai_analyst() -> None:
    st.markdown(_AI_CSS, unsafe_allow_html=True)

    # Status line (compact — page title already shows "AI Analyst")
    st.markdown(
        '<div style="display:flex;align-items:center;gap:8px;margin-bottom:16px">'
        '<span class="ai-dot"></span>'
        f'<span style="font-family:IBM Plex Mono,monospace;font-size:12px;color:{CHART_AXIS};letter-spacing:1px">'
        'LLAMA-3.3-70B-VERSATILE · GROQ · 11 TOOLS</span></div>',
        unsafe_allow_html=True,
    )

    # ── Session state ────────────────────────────────────────────────────────────
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "ai_agent" not in st.session_state:
        st.session_state.ai_agent = None
    if "agent_error" not in st.session_state:
        st.session_state.agent_error = None

    # ── Build agent (once) ───────────────────────────────────────────────────────
    if st.session_state.ai_agent is None and st.session_state.agent_error is None:
        with st.spinner("Loading AI Analyst…"):
            try:
                st.session_state.ai_agent = _build_agent()
            except Exception as exc:
                st.session_state.agent_error = str(exc)

    if st.session_state.agent_error:
        st.error(f"Could not load agent: {st.session_state.agent_error}")
        st.info("Check that `GROQ_API_KEY` is set in `.env` and the pipeline has been run.")
        return

    agent = st.session_state.ai_agent

    # ── Suggested prompt chips ───────────────────────────────────────────────────
    if not st.session_state.chat_history:
        st.markdown('<div class="chip-row">', unsafe_allow_html=True)
        cols = st.columns(len(_SUGGESTED))
        for i, suggestion in enumerate(_SUGGESTED):
            with cols[i]:
                st.markdown('<div class="chip-btn">', unsafe_allow_html=True)
                if st.button(suggestion, key=f"chip_{i}", use_container_width=True):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Pre-filled prompt from Deep-Dive ─────────────────────────────────────────
    if st.session_state.get("ai_prefill"):
        prefill = st.session_state.pop("ai_prefill")
        st.session_state.chat_history.append({"role": "user", "content": prefill})

    # ── Render existing chat history ─────────────────────────────────────────────
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                tools = msg.get("tools", [])
                if tools:
                    chips = " &nbsp;".join(
                        f'<span class="tool-chip">used: {t}</span>' for t in tools
                    )
                    st.markdown(f'<div class="tool-chips">{chips}</div>', unsafe_allow_html=True)
            st.markdown(msg["content"])

    # ── Process pending user message (no AI response yet) ───────────────────────
    pending  = [m for m in st.session_state.chat_history if m["role"] == "user"]
    answered = [m for m in st.session_state.chat_history if m["role"] == "assistant"]
    if len(pending) > len(answered):
        prompt = pending[-1]["content"]
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, tool_names = _run_agent(agent, prompt)
            # Tool chips
            if tool_names:
                chips = " &nbsp;".join(
                    f'<span class="tool-chip">used: {t}</span>' for t in tool_names
                )
                st.markdown(f'<div class="tool-chips">{chips}</div>', unsafe_allow_html=True)
            st.markdown(answer)

        # Store both tools and answer so re-renders work
        record = {"role": "assistant", "content": answer, "tools": tool_names}
        st.session_state.chat_history.append(record)
        st.rerun()

    # ── Chat input ───────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask about any stock, sector, or risk metric…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        st.rerun()

    # ── Controls ─────────────────────────────────────────────────────────────────
    if st.session_state.chat_history:
        if st.button("Clear conversation", key="clear_chat"):
            st.session_state.chat_history = []
            st.rerun()

    st.markdown(
        '<div class="ai-disclaimer">FinWatch AI provides decision-support only — not financial advice. '
        "Regulatory constraint: the agent will not recommend buy, sell, or hold actions.</div>",
        unsafe_allow_html=True,
    )
