import datetime
from pathlib import Path
from collections import defaultdict
import streamlit as st
import streamlit.components.v1 as components
from enochian_translation_team.crew import run_crew  # Import here, safely
# Placeholder for future import!! We will get there...
# from enochian_translation_team.tools.extract_roots import run_root_extraction

token_buffers = defaultdict(str)
placeholders = {}

def save_log_to_txt():
    if "log" not in st.session_state or not st.session_state.log:
        st.warning("🫥 Nothing to log.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("src/enochian_translation_team/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{timestamp}_log.txt"

    with open(log_path, "w", encoding="utf-8") as f:
        for role, message in st.session_state.log:
            f.write(f"{role}: {message.strip()}\n\n")

    st.success(f"📁 Log saved to `{log_path}`")

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
st.markdown(custom_css, unsafe_allow_html=True)
st.markdown("Because why not look at glossolalia with a fresh set of AIs.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
dictionary_percent = st.sidebar.slider("Percent of dictionary to process:", 1, 100, 100)

# --- Callback Function ---
def stream_callback(role, message):
    # Set up emoji and display name
    emoji_map = {
        "Computational Linguist": "💻",
        "Adjudicator": "👩‍⚖️",
        "Skeptic": "🤔",
        "Archivist": "📚",
        "Maestro": "🪄"
    }
    badge = emoji_map.get(role, "👤")
    display_name = f"{badge} {role}"

    # Create message box only once
    if role not in placeholders:
        with st.chat_message(display_name, avatar=badge):
            placeholders[role] = st.empty()
        token_buffers[role] = ""

    # Append token and update display
    token_buffers[role] += message
    placeholders[role].markdown(token_buffers[role])

# --- Button to Trigger Agent Task ---
if st.sidebar.button("🧠 Extract Root Words"):
    st.session_state.log = []

    st.markdown("### 💬 Agent Chat Log")


    with st.chat_message("Maestro", avatar="🪄"):
        st.markdown("_**Initializing semantic tribunal...**_")

    # TEMP: hardcoded word, definition
    run_crew(word="AAI", definition="amongst", stream_callback=stream_callback)

    st.success("Crew has completed their assigned task. Hooray. 🎉")

    for role, text in token_buffers.items():
            st.session_state.log.append((role, text))
    save_log_to_txt()