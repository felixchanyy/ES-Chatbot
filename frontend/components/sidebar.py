# frontend/components/sidebar.py

import streamlit as st
import requests

def render_sidebar(backend_url: str):
    st.header("Index Stats")
    if st.button("Refresh Stats"):
        st.rerun()
    
    # Fetch index stats from the backend
    stats_url = f"{backend_url}/api/v1/index/stats"
    try:
        response = requests.get(stats_url)
        if response.status_code == 200:
            stats = response.json()
            st.write(f"Total Documents: {stats.get('total_documents', 'N/A')}")
            st.write(f"Index Size: {stats.get('index_size_bytes', 'N/A')} bytes")
            st.write(f"Earliest Date: {stats.get('earliest_date', 'N/A')}")
            st.write(f"Latest Date: {stats.get('latest_date', 'N/A')}")
        else:
            st.error("Failed to fetch index stats.")
    except Exception as e:
        st.error(f"Error fetching index stats: {e}")
    
    st.header("Settings")
    st.checkbox("Show raw query", key="show_raw_query")
    st.button("Clear Chat", on_click=lambda: st.session_state.update(messages=[]))