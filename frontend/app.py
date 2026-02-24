# frontend/app.py

import streamlit as st
import requests
from components.sidebar import render_sidebar
from components.chat import render_chat_history, render_chat_input
import uuid  # Add this import

BACKEND_URL = "http://backend:8000"

def main():
    st.set_page_config(page_title="GKG OSINT Chatbot", layout="wide")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Add unique session tracking
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    with st.sidebar:
        render_sidebar(backend_url=BACKEND_URL)

    render_chat_history(st.session_state.messages)
    render_chat_input(backend_url=BACKEND_URL)

if __name__ == "__main__":
    main()