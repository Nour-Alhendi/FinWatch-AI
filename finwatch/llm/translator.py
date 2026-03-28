"""
FinWatch AI — LLM Translator
==============================
Translates LLM summaries to German or Arabic via Groq.
Cached in Streamlit session state per ticker+language pair.
"""

import os
import streamlit as st


def translate(text: str, language: str, ticker: str) -> str:
    """Translate llm_summary to target language via Groq. Cached in session state."""
    if language == "english":
        return text
    if "llm_cache" not in st.session_state:
        st.session_state.llm_cache = {}
    cache_key = f"{ticker}_{language}"
    if cache_key in st.session_state.llm_cache:
        return st.session_state.llm_cache[cache_key]
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return text
    lang_map = {"german": "German", "arabic": "Arabic"}
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content":
                f"Translate the following financial analysis to {lang_map[language]}. "
                f"Keep all section headers (**bold**), numbers, and formatting exactly. "
                f"Only translate the text, nothing else.\n\n{text}"}],
            temperature=0.1,
            max_tokens=1800,
        )
        result = resp.choices[0].message.content.strip()
        st.session_state.llm_cache[cache_key] = result
        return result
    except Exception as e:
        st.warning(f"Translation failed: {e}")
        return text
