import streamlit as st
import requests
import json
from stream_service.producer import stream_rlhf_feedback
import os
import base64
import streamlit.components.v1 as components
import time

willi_avatar = "😇"

st.set_page_config(page_title="WiLLi", layout="wide", page_icon=willi_avatar)

# --- Custom Styling for WiLLi ---
st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    /* Hide Streamlit's default running animation */
    [data-testid="stStatusWidget"] {{
        display: none;
    }}
    
    .stDeployButton {{
        display: none;
    }}

    .main-header {{
        font-family: 'Poppins', sans-serif;
        font-size: 3rem; 
        font-weight: 600;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 0px;
        letter-spacing: -0.5px;
    }}
    .sub-header {{
        font-family: 'Poppins', sans-serif;
        text-align: center;
        color: #666;
        font-weight: 400;
        margin-bottom: 2rem;
    }}
    .stApp {{
        transition: none !important;
        opacity: 1 !important;
    }}
    </style>
    <div class="main-header">WiLLi</div>
    <div class="sub-header">Will's AI Clone {willi_avatar}</div>
""", unsafe_allow_html=True)

# --- Sidebar Configuration & Auth ---
st.sidebar.title("Mode Control")
mode = st.sidebar.radio(
    "Select Environment:",
    ["Public", "Admin (DPO RL Training Mode)"],
    help="Admin mode enables real-time data streaming to Redpanda for DPO optimization."
)

# Initialize admin state
if "admin_authenticated" not in st.session_state:
    st.session_state.admin_authenticated = False

if mode == "Admin (DPO RL Training Mode)":
    if not st.session_state.admin_authenticated:
        st.sidebar.markdown("---")
        password = st.sidebar.text_input("Admin Password", type="password")
        if st.sidebar.button("Login"):
            if password == st.secrets["ADMIN_PASSWORD"]: 
                st.session_state.admin_authenticated = True
                st.rerun()
            else:
                st.sidebar.error("Invalid Credentials")
    
    if st.session_state.admin_authenticated:
        st.sidebar.success("RLHF Data Stream: ACTIVE")
        st.sidebar.info("Model: Qwen 2.5 1.5B (CUDA Accelerated)")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Training Logs")
        log_file = os.path.join(os.getcwd(), "training_logs.txt")
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                logs = f.readlines()
                log_text = "".join(logs[-10:])
                st.sidebar.code(log_text, language="text")
        else:
            st.sidebar.caption("No training events logged yet.")
            
        if st.sidebar.button("Logout"):
            st.session_state.admin_authenticated = False
            st.rerun()
else:
    st.session_state.admin_authenticated = False
    st.sidebar.warning("Public View")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs(["Chat with WiLLi", "Architecture", "Projects", "Resume"])

with tab1:
    # --- Chat Logic ---
    if "messages" not in st.session_state:
        intro_1 = "Hi! I'm WiLLi, Will's AI clone. Feel free to ask me anything about my career, projects, or technical skills!"
        intro_2 = "Admins can log in to activate real-time DPO training so my responses improve instantly."
        
        st.session_state.messages = [
            {"role": "assistant", "content": intro_1, "avatar": willi_avatar},
            {"role": "assistant", "content": intro_2, "avatar": willi_avatar}
        ]
        
    # Display chat history
    chat_container = st.container(height=500)
    with chat_container:
        for message in st.session_state.messages:
            current_avatar = message.get("avatar") if message["role"] == "assistant" else None
            with st.chat_message(message["role"], avatar=current_avatar):
                st.markdown(message["content"])

    # User Input
    chat_placeholder = "Ask me about my career..." if mode == "Public" else "ENTER PROMPT FOR DPO OPTIMIZATION..."
    if prompt := st.chat_input(chat_placeholder):
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            response = requests.post(
                "http://localhost:8000/chat", 
                json={"question": prompt},
                stream=True  
            )
            
            def stream_parser():
                for chunk in response.iter_content(chunk_size=16, decode_unicode=True):
                    if chunk:
                        yield chunk

            with chat_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant", avatar=willi_avatar):
                    answer = st.write_stream(stream_parser())

            st.session_state.last_prompt = prompt
            st.session_state.last_answer = answer
            st.session_state.messages.append({"role": "assistant", "content": answer, "avatar": willi_avatar})

        except Exception as e:
            st.error(f"Failed to connect to WiLLi Backend: {e}")

    # --- ADMIN RLHF SECTION ---
    if mode == "Admin (DPO RL Training Mode)":
        if not st.session_state.admin_authenticated:
            st.warning("Please log in via the sidebar to access Admin features.")
        else:
            st.write("---")
            st.subheader("RLHF Feedback Loop and DPO Ingestion")
            st.caption("Submitting feedback here streams data directly to Redpanda for real-time model alignment.")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Log as Preferred Response"):
                    p = st.session_state.get("last_prompt")
                    a = st.session_state.get("last_answer")
                    if p and a:
                        stream_rlhf_feedback(p, a, a)
                        st.success("Signal sent: Positive Reward Logged.")
            
            with col2:
                correction = st.text_input("Correct this response (Will's Voice):")
                if st.button("Push to Redpanda for Training"):
                    p = st.session_state.get("last_prompt")
                    a = st.session_state.get("last_answer")
                    if correction and p and a:
                        stream_rlhf_feedback(p, correction, a)
                        st.warning(f"DPO Pair Ingested: '{correction[:30]}...'")
                    else:
                        st.error("No active chat context to train on!")

            status_file = os.path.join(os.getcwd(), "training_status.json")
            if os.path.exists(status_file):
                with open(status_file, "r") as f:
                    s = json.load(f)
                if s.get("status") == "training":
                    st.progress(0.7, text="DPO training in progress...")
                elif s.get("status") == "complete":
                    st.progress(1.0, text="Training complete — adapter saved.")

            st.markdown("Last Training Step")
            metrics_file = os.path.join(os.getcwd(), "training_metrics.json")

            col_refresh, col_auto, _ = st.columns([1, 2, 3])
            with col_refresh:
                if st.button("↻ Refresh"):
                    st.rerun()
            with col_auto:
                auto = st.toggle("Auto-refresh (2s)", value=True)

            if os.path.exists(metrics_file):
                with open(metrics_file, "r", encoding="utf-8") as f:
                    m = json.load(f)

                st.caption(f"Last update: {m.get('timestamp', 'N/A')}")

                with st.expander("Training Pair Preview", expanded=True):
                    st.markdown(f"**Prompt:** `{m.get('prompt_preview')}...`")
                    st.markdown(f"**Chosen:** `{m.get('chosen_preview')}...`")
                    st.markdown(f"**Rejected:** `{m.get('rejected_preview')}...`")

                def fmt(key, decimals=4):
                    v = m.get(key)
                    try:
                        return round(float(v), decimals)
                    except (TypeError, ValueError):
                        return "—"

                m1, m2, m3 = st.columns(3)
                m1.metric("Loss", fmt('loss'))
                m2.metric("Grad Norm", fmt('grad_norm'))
                m3.metric("Reward Margin", fmt('rewards_margin'))

                m4, m5, m6 = st.columns(3)
                m4.metric("Accuracy", fmt('rewards_accuracy'))
                m5.metric("LogP Chosen", fmt('logps_chosen', 2))
                m6.metric("LogP Rejected", fmt('logps_rejected', 2))

                try:
                    margin = float(m.get('rewards_margin') or 0)
                    grad = float(m.get('grad_norm') or 0)
                except (TypeError, ValueError):
                    margin, grad = 0, 0

                if margin > 0 and grad > 1:
                    st.success("Strong training signal — model updated meaningfully.")
                elif grad > 0.01:
                    st.warning("Weak signal — model nudged slightly. Feed more correction pairs.")
                else:
                    st.error("Zero signal — chosen and rejected were identical. Use the correction box.")
            else:
                st.caption("No training steps logged yet. Submit a correction to trigger training.")

            if auto:
                time.sleep(5)
                st.rerun()

with tab2:
    if mode == "Admin (DPO RL Training Mode)":
        st.info("Switch to **Public** mode to view Architecture.")
    else:
        with open("architecture.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=850, scrolling=True)
    
with tab3:
    if mode == "Admin (DPO RL Training Mode)":
        st.info("Switch to **Public** mode to view Projects.")
    else:
        with open("projects.html", "r", encoding="utf-8") as f:
            html = f.read()
        components.html(html, height=900, scrolling=True)
    
with tab4:
    
    if mode == "Admin (DPO RL Training Mode)":
        st.info("Switch to **Public** mode to view Resume.")
    else:
        if os.path.exists("resume.pdf"):
            with open("resume.pdf", "rb") as f:
                pdf_bytes = f.read()
            st.download_button("Download Resume", pdf_bytes, "Will_Resume.pdf", "application/pdf")
            b64 = base64.b64encode(pdf_bytes).decode()
            st.markdown(f'<iframe src="data:application/pdf;base64,{b64}" width="100%" height="800px"></iframe>', unsafe_allow_html=True)
        else:
            st.error("resume.pdf not found in project root.")