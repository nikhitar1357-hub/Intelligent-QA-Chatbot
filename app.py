"""
app.py - Streamlit front-end for the Intelligent QA / Chatbot System.

Run with:
    streamlit run app.py
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.chatbot import Chatbot, MODE_DIALOGUE, MODE_KB, MODE_IR, MODE_FALLBACK

st.set_page_config(page_title="Intelligent QA Chatbot", page_icon="🤖", layout="wide")

MODE_BADGE = {
    MODE_DIALOGUE: ("💬 Dialogue", "#6c757d"),
    MODE_KB: ("📚 Knowledge QA", "#2e7d32"),
    MODE_IR: ("🔎 IR QA", "#1565c0"),
    MODE_FALLBACK: ("⚠️ Fallback", "#b71c1c"),
}

# ---------------------------------------------------------------- session --
if "bot" not in st.session_state:
    st.session_state.bot = Chatbot()
if "chat" not in st.session_state:
    st.session_state.chat = []   # list of dicts: role, text, mode, confidence, context, source

bot: Chatbot = st.session_state.bot

# ----------------------------------------------------------------- sidebar --
with st.sidebar:
    st.title("🤖 QA Chatbot")
    st.caption("IR-based QA · Knowledge-based QA · Dialogue Management")

    st.markdown("### ℹ️ About")
    st.write(
        "A project demonstrating three QA techniques: "
        "TF-IDF information retrieval over text documents, structured "
        "knowledge-base lookup, and rule-based dialogue management with memory."
    )

    st.markdown("### 🧩 Knowledge sources")
    st.write(f"- **{len(bot.ir.passages)}** text passages across 20 documents")
    st.write(f"- **{len(bot.kb.entries)}** structured facts across {len(bot.kb.categories())} categories")
    st.write(", ".join(bot.kb.categories()))

    st.markdown("### 💡 Try asking")
    examples = [
        "Hi, how are you?",
        "What is machine learning?",
        "Who invented Python?",
        "What is the capital of Karnataka?",
        "How does backpropagation work?",
        "What is overfitting?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True, key=f"ex_{ex}"):
            st.session_state.pending_input = ex

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.chat = []
        bot.reset()
        st.rerun()

# ------------------------------------------------------------------- main --
st.markdown("## Intelligent Question Answering & Chatbot System")
st.caption("Information Retrieval · Knowledge-Based QA · Dialogue Management")

chat_box = st.container()


def render_message(msg):
    if msg["role"] == "user":
        with chat_box.chat_message("user"):
            st.write(msg["text"])
    else:
        with chat_box.chat_message("assistant"):
            label, color = MODE_BADGE.get(msg["mode"], ("🤖", "#555"))
            st.markdown(
                f"<span style='background:{color};color:white;padding:2px 10px;"
                f"border-radius:12px;font-size:0.78em'>{label}</span>",
                unsafe_allow_html=True,
            )
            st.write(msg["text"])
            if msg["mode"] in (MODE_KB, MODE_IR):
                cols = st.columns([1, 1])
                cols[0].progress(min(1.0, msg["confidence"]), text=f"Confidence: {msg['confidence']:.0%}")
                cols[1].caption(f"⏱ {msg['elapsed']*1000:.1f} ms  ·  source: {msg['source']}")
                if msg.get("context"):
                    with st.expander("Show retrieved context"):
                        st.write(msg["context"])


for m in st.session_state.chat:
    render_message(m)

prefill = st.session_state.pop("pending_input", "")
user_text = st.chat_input("Ask a question or just say hi...")
if prefill and not user_text:
    user_text = prefill

if user_text:
    st.session_state.chat.append({"role": "user", "text": user_text})
    with st.spinner("Thinking..."):
        resp = bot.ask(user_text)
    st.session_state.chat.append({
        "role": "bot", "text": resp.text, "mode": resp.mode,
        "confidence": resp.confidence, "context": resp.context,
        "source": resp.source, "elapsed": resp.elapsed,
    })
    st.rerun()

if not st.session_state.chat:
    st.info("👋 Start the conversation using the chat box below, or click an example in the sidebar.")
