import streamlit as st
import requests
import os

API_URL = os.getenv("API_URL", "https://knot-ai.site/api/v1")

st.set_page_config(
    page_title="Wedding Agent",
    page_icon="💍",
    layout="centered",
)

st.title("💍 Wedding Vendor Assistant")
st.caption("Find venues, caterers, photographers & more across India. Ask in English or Hinglish.")

# ── session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ── sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Examples")
    examples = [
        "Suggest venues in Mumbai for 200 guests with veg food",
        "Wedding photographer in Hyderabad under ₹50000",
        "Pandit for haldi and mehndi ceremony in Delhi",
        "Caterers for sangeet and baarat in Jaipur",
        "Compare photographers in Kolkata",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state.pending_input = ex

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = None
        st.rerun()

# ── chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input("Ask about wedding vendors...")

# handle example button click
if "pending_input" in st.session_state:
    user_input = st.session_state.pop("pending_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Searching vendors..."):
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
                answer = "Request timed out. Please try again."
            except Exception as e:
                answer = f"Error: {e}"

        st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
