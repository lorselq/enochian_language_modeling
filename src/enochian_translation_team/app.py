# src/enochian_translation_team/app.py

import streamlit as st
from enochian_translation_team.crew import run_crew  # Import here, safely
# Placeholder for future import!! We will get there...
# from enochian_translation_team.tools.extract_roots import run_root_extraction

custom_css = """
<style>
    body {
        background-color: #121212;
        color: #e0e0e0;
        font-family: 'Georgia', serif;
    }

    .stChatMessage {
        border-radius: 12px;
        padding: 0.5rem 1rem;
        margin-bottom: 1rem;
        font-size: 0.95rem;
        box-shadow: 0 0 6px rgba(255, 215, 0, 0.05);
    }

    .stChatMessage[data-testid="chat-message-orchestrator"] {
        background-color: rgba(255, 228, 181, 0.07);
        border-left: 4px solid #FFD700;
    }

    .stChatMessage[data-testid="chat-message-linguist"] {
        background-color: rgba(135, 206, 250, 0.06);
        border-left: 4px solid #87CEFA;
    }

    .stChatMessage[data-testid="chat-message-skeptic"] {
        background-color: rgba(240, 128, 128, 0.07);
        border-left: 4px solid #FA8072;
    }

    .stChatMessage:hover {
        background-color: rgba(255, 255, 255, 0.05);
        transition: 0.3s ease;
    }

    .stButton>button {
        background-color: #3a3a3a;
        color: #f1f1f1;
        border: 1px solid #888;
        border-radius: 8px;
    }

    .stButton>button:hover {
        background-color: #444;
        border: 1px solid #FFD700;
        color: #FFD700;
    }

    .stSlider>div>div>div {
        background-color: #FFD700 !important;
    }

    .stSidebar {
        background-color: #1e1e1e;
        color: #e0e0e0;
    }

    .stSidebar h2 {
        color: #FFD700;
        font-family: 'Georgia', serif;
    }

    h1, h2, h3, h4 {
        font-family: 'Georgia', serif;
        color: #FFD700;
    }

    .stMarkdown {
        font-family: 'Georgia', serif;
    }
</style>
"""
# --- Streamlit Setup ---
st.set_page_config(page_title="Enochian Language Modeling Interface", layout="wide")
st.title("📜 Enochian Language Modeling Interface")
st.markdown("Because why not look at glossolalia with a fresh set of AIs.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
dictionary_percent = st.sidebar.slider("Percent of dictionary to process:", 1, 100, 100)

st.markdown(custom_css, unsafe_allow_html=True)

# --- Callback Function ---
def stream_callback(role, message):
    if "log" not in st.session_state:
        st.session_state.log = []

    # Assign emoji badge
    emoji_map = {
        "Computational Linguist": "💻",
        "Adjudicator": "👩‍⚖️",
        "Skeptic": "🤔",
        "Archivist": "📚"
    }

    badge = emoji_map.get(role, "👤")
    display_role = f"{badge} {role}"

    st.session_state.log.append((display_role, message))
    with st.chat_message(display_role):
        st.markdown(message)

# --- Button to Trigger Agent Task ---
if st.sidebar.button("🧠 Extract Root Words"):
    st.session_state.log = []

    st.markdown("### 💬 Agent Chat Log")

    with st.chat_message("🪄 Maestro"):
        st.markdown("_Initializing semantic tribunal..._")

    # TEMP: hardcoded word, definition
    run_crew(word="AAI", definition="amongst", stream_callback=stream_callback)

    st.success("Crew has completed their assigned task. Hooray.")

# --- Log re-display (e.g., page refresh) ---
if "log" in st.session_state and st.session_state.log:
    st.markdown("### 💬 Agent Chat Log (History)")
    for role, message in st.session_state.log:
        with st.chat_message(role):
            st.markdown(message)