"""
Persistent AI Analyst chat panel — shown alongside Deep-Dive and Command Center.
"""
from __future__ import annotations
import html
import sys
from pathlib import Path
import streamlit as st

from data.loader import COMPANY_NAMES
from ui.i18n import t
from ui.theme import TEXT_MUTED

# Set to False to fully hide tool-call disclosures in the chat panel.
SHOW_TOOL_CALLS = True

_CSS = """
<style>
/* ── Panel header ── */
.cp-header{
    display:flex;align-items:center;gap:8px;
    padding-bottom:10px;
    border-bottom:1px solid rgba(255,255,255,0.14);
    margin-bottom:6px;
}
.cp-dot{
    width:7px;height:7px;border-radius:50%;
    background:#00c4b4;display:inline-block;flex-shrink:0;
    box-shadow:0 0 6px rgba(0,196,180,0.35);
}
.cp-title{
    font-family:'Inter',sans-serif;
    font-size:14px;font-weight:500;color:#f0f0f5;
}
.cp-meta{
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;color:#4a4a5a;margin-left:auto;
}
.cp-context{
    font-family:'Inter',sans-serif;
    font-size:12px;color:#4a4a5a;font-style:italic;
    margin-bottom:8px;
}

/* ── Hint header ── */
.cp-hint-hdr{
    font-family:'IBM Plex Mono',monospace;
    font-size:11px;letter-spacing:0.08em;text-transform:uppercase;
    color:#4a4a5a;padding:8px 0 10px;
}

/* ── Suggestion chips ── */
[data-testid="element-container"]:has(.cp-chips-mark) ~ [data-testid="element-container"] [data-testid="stButton"] button{
    background:transparent!important;
    border:1px solid rgba(255,255,255,0.18)!important;
    color:#8b8b9e!important;
    font-family:'Inter',sans-serif!important;
    font-size:13px!important;
    border-radius:8px!important;
    padding:10px 14px!important;
    min-height:40px!important;
    text-align:left!important;
    justify-content:flex-start!important;
    width:100%!important;
    line-height:1.4!important;
    white-space:normal!important;
    transition:border-color 0.15s,color 0.15s!important;
    box-shadow:none!important;
}
[data-testid="element-container"]:has(.cp-chips-mark) ~ [data-testid="element-container"] [data-testid="stButton"] button:hover{
    border-color:#00c4b4!important;
    color:#f0f0f5!important;
}

/* ── User bubble (right-aligned) ── */
.cp-user-msg{
    background:rgba(0,196,180,0.1);
    border:1px solid rgba(0,196,180,0.2);
    border-radius:12px;
    padding:10px 14px;
    margin-left:20%;
    margin-bottom:8px;
    font-family:'Inter',sans-serif;
    font-size:13px;color:#f0f0f5;
    line-height:1.6;
    word-break:break-word;
}

/* ── Agent bubble (left-aligned via st.chat_message) ── */
[data-testid="stChatMessage"]{
    background:#16161f!important;
    border:1px solid rgba(255,255,255,0.14)!important;
    border-radius:12px!important;
    padding:10px 14px!important;
    margin-right:20%!important;
    margin-bottom:0!important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] code{
    font-size:13px!important;
    line-height:1.6!important;
}

/* ── Tool call disclosure (collapsible) ── */
.tc-wrap{margin-bottom:4px;user-select:none}
.tc-summary{
    font-size:11px;color:#4a4a5a;cursor:pointer;
    font-family:'IBM Plex Mono',monospace;
    list-style:none;outline:none;
    display:inline-block;
}
.tc-summary::-webkit-details-marker{display:none}
.tc-summary:hover{color:#6b6b7e}
.tc-list{padding:3px 0 2px 10px}
.tc-name{
    display:block;font-size:10px;
    color:#4a4a5a;font-family:'IBM Plex Mono',monospace;
    line-height:1.8;
}

/* ── Form input ── */
[data-testid="stForm"]{border:none!important;padding:0!important;background:transparent!important}
[data-testid="stTextInput"] input{
    background:#111118!important;
    border:1px solid rgba(255,255,255,0.15)!important;
    color:#f0f0f5!important;
    font-family:'Inter',sans-serif!important;
    font-size:13px!important;
    border-radius:8px!important;
    padding:10px 14px!important;
}
[data-testid="stTextInput"] input:focus{
    border-color:rgba(0,196,180,0.4)!important;
    box-shadow:0 0 0 1px rgba(0,196,180,0.1)!important;
    outline:none!important;
}
[data-testid="stTextInput"] input::placeholder{color:#4a4a5a!important}
[data-testid="InputInstructions"]{display:none!important}

/* ── Submit button ── */
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button{
    background:rgba(0,196,180,0.12)!important;
    border:1px solid rgba(0,196,180,0.25)!important;
    color:#00c4b4!important;
    border-radius:8px!important;
    font-size:16px!important;
    font-weight:500!important;
    height:100%!important;
    min-height:42px!important;
}
[data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover{
    background:rgba(0,196,180,0.20)!important;
    border-color:#00c4b4!important;
}

/* ── Scrollable container border ── */
[data-testid="stVerticalBlockBorderWrapper"]{
    border-color:rgba(255,255,255,0.14)!important;
    border-radius:8px!important;
    background:rgba(10,10,15,0.3)!important;
}
</style>
"""
_CSS = (
    _CSS
    .replace("#4a4a5a", TEXT_MUTED)
    .replace("#6b6b7e", TEXT_MUTED)
)


def _ensure_agent() -> bool:
    """Build the agent on first call. Returns True if ready."""
    if st.session_state.ai_agent is not None:
        return True
    if st.session_state.agent_error:
        return False

    with st.spinner("Loading AI Analyst…"):
        try:
            _root = Path(__file__).resolve().parents[3]
            if str(_root) not in sys.path:
                sys.path.insert(0, str(_root))
            from src.agent.agent import build_agent  # noqa: PLC0415
            st.session_state.ai_agent = build_agent()
            return True
        except Exception as exc:
            import logging
            logging.getLogger(__name__).error(
                "Agent init failed (%s): %s", type(exc).__name__, exc, exc_info=True
            )
            msg = str(exc).lower()
            if "groq_api_key" in msg or "api_key" in msg or "key" in msg or "auth" in msg:
                st.session_state.agent_error = "missing_key"
            else:
                st.session_state.agent_error = "init_failed"
            return False


def render_chat_panel(ticker: str | None = None) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)

    company   = COMPANY_NAMES.get(ticker, ticker) if ticker else None
    scope_key = ticker or "portfolio"
    portfolio_lbl = t("portfolio_label")

    if st.session_state.get("panel_chat_context") != scope_key:
        st.session_state.panel_chat_context = scope_key
        st.session_state.panel_chat_history = []

    # Header
    st.markdown(
        '<div class="cp-header">'
        '<span class="cp-dot"></span>'
        '<span class="cp-title">AI Analyst</span>'
        '<span class="cp-meta">11 tools · Groq LLaMA 3.3 70B</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Context line
    if ticker and company:
        st.markdown(
            f'<div class="cp-context">Viewing {html.escape(company)} ({html.escape(ticker)})</div>',
            unsafe_allow_html=True,
        )

    if not _ensure_agent():
        _err = st.session_state.agent_error
        if _err == "missing_key":
            st.warning("⚠️ AI Analyst is unavailable — API key not configured.")
            st.caption("Set `GROQ_API_KEY` in your `.env` file and restart the app.")
        else:
            st.warning("⚠️ AI Analyst failed to load. Please restart the app.")
        return

    history: list[dict] = st.session_state.panel_chat_history

    hints = (
        [t("chat_hint_why"), t("chat_hint_risk"), t("chat_hint_analysts"), t("chat_hint_trend")]
        if ticker else
        [t("chat_hint_portfolio_risk"), t("chat_hint_sector"), t("chat_hint_overview"), t("chat_hint_entry")]
    )

    # ── Scrollable message area ───────────────────────────────────────────────
    with st.container(height=560, border=True):
        if not history:
            subject = company or portfolio_lbl
            st.markdown(
                f'<div class="cp-hint-hdr">{t("chat_about", subject=subject)}</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<span class="cp-chips-mark"></span>', unsafe_allow_html=True)
            for i, hint in enumerate(hints):
                if st.button(hint, key=f"cp_hint_{i}", use_container_width=True):
                    st.session_state.panel_chat_history.append({"role": "user", "content": hint})
                    st.rerun()
        else:
            for msg in history:
                if msg["role"] == "user":
                    st.markdown(
                        f'<div class="cp-user-msg">{html.escape(msg["content"])}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    tool_names = msg.get("tools", [])
                    if tool_names and SHOW_TOOL_CALLS:
                        n = len(tool_names)
                        names_html = "".join(f'<span class="tc-name">⚙ {tn}</span>' for tn in tool_names)
                        st.markdown(
                            f'<details class="tc-wrap"><summary class="tc-summary">'
                            f'▸ {n} tool{"s" if n != 1 else ""} used'
                            f'</summary><div class="tc-list">{names_html}</div></details>',
                            unsafe_allow_html=True,
                        )
                    with st.chat_message("assistant"):
                        st.markdown(msg["content"])

            # Run agent for any unanswered user message
            pending  = [m for m in history if m["role"] == "user"]
            answered = [m for m in history if m["role"] == "assistant"]
            if len(pending) > len(answered):
                user_text = pending[-1]["content"]
                agent_prompt = (
                    f"The user is viewing {ticker} ({company}) in the Stock Deep-Dive. "
                    f"Answer the following, defaulting to {ticker} unless another stock is explicitly named: "
                    f"{user_text}"
                ) if ticker else (
                    f"The user is on the Command Center (portfolio overview). {user_text}"
                )

                _lang = st.session_state.get("lang", "en")
                with st.spinner("Thinking…"):
                    try:
                        answer, tool_names = st.session_state.ai_agent.run(agent_prompt, lang=_lang)
                    except Exception as exc:
                        import logging
                        logging.getLogger(__name__).error(
                            "Chat panel unexpected error: %s", exc, exc_info=True
                        )
                        answer, tool_names = (
                            "⚠️ Something went wrong while generating the answer. "
                            "Please rephrase or try again.",
                            [],
                        )

                if tool_names and SHOW_TOOL_CALLS:
                    n = len(tool_names)
                    names_html = "".join(f'<span class="tc-name">⚙ {tn}</span>' for tn in tool_names)
                    st.markdown(
                        f'<details class="tc-wrap"><summary class="tc-summary">'
                        f'▸ {n} tool{"s" if n != 1 else ""} used'
                        f'</summary><div class="tc-list">{names_html}</div></details>',
                        unsafe_allow_html=True,
                    )
                with st.chat_message("assistant"):
                    st.markdown(answer)

                st.session_state.panel_chat_history.append({
                    "role": "assistant", "content": answer, "tools": tool_names,
                })
                st.rerun()

    # ── Form input ───────────────────────────────────────────────────────────
    placeholder = t("chat_ask_stock", ticker=ticker) if ticker else t("chat_ask_portfolio")
    with st.form(key="cp_chat_form", clear_on_submit=True):
        col_in, col_btn = st.columns([6, 1])
        with col_in:
            user_input = st.text_input("", placeholder=placeholder, label_visibility="collapsed")
        with col_btn:
            submitted = st.form_submit_button("→", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.panel_chat_history.append({"role": "user", "content": user_input.strip()})
        st.rerun()
