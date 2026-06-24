import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "https://knot-ai.site/api/v1")

st.set_page_config(
    page_title="AI Wedding Planner",
    page_icon="💍",
    layout="centered",
)

st.title("💍 AI Wedding Planner")
st.caption("Plan your dream Indian wedding — vendors, budgets, rituals, and more. Ask in English or Hinglish.")

# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Start a conversation")
    st.markdown("Try asking anything about your wedding:")

    example_groups = {
        "Find Vendors": [
            "Suggest venues in Mumbai for 200 guests with veg food",
            "Wedding photographer in Hyderabad under ₹50000",
            "Caterers for sangeet and baarat in Jaipur",
        ],
        "Plan & Budget": [
            "How much will a 300 guest wedding cost in Delhi?",
            "What's a typical wedding timeline for a 2-day ceremony?",
            "What rituals are part of a North Indian wedding?",
        ],
    }

    for group, examples in example_groups.items():
        st.markdown(f"**{group}**")
        for ex in examples:
            if st.button(ex, use_container_width=True, key=ex):
                st.session_state.pending_input = ex

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# ── welcome message ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "Namaste! I'm your AI wedding planner. I can help you:\n\n"
            "- **Find vendors** — venues, caterers, photographers, pandits, decorators & more\n"
            "- **Plan your budget** — real price estimates based on your city and guest count\n"
            "- **Answer planning questions** — rituals, timelines, traditions, Hinglish terms\n\n"
            "What are you planning for your wedding? 🎊"
        )

# ── chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me anything about your wedding...")

if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/chat",
                    json={
                        "message":    user_input,
                        "session_id": st.session_state.session_id,
                    },
                    timeout=60,
                )
                resp.raise_for_status()
                data = resp.json()
                answer = data["answer"]
                st.session_state.session_id = data["session_id"]
            except requests.exceptions.Timeout:
                answer = "Taking longer than usual — please try again."
            except Exception as e:
                answer = f"Something went wrong: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
