# frontend/components/chat.py

import streamlit as st
import requests

def render_chat_history(messages: list[dict]):
    """Render the chat history."""
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

def render_chat_input(backend_url: str):
    """Render the chat input and handle user messages."""
    if prompt := st.chat_input("Type your question..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            
            # Send user message to backend
            chat_url = f"{backend_url}/api/v1/chat"
            payload = {
                "message": prompt,
                "session_id": st.session_state.session_id, # Use the dynamic ID
                "history": st.session_state.messages[:-1]
            }
            
            try:
                response = requests.post(chat_url, json=payload)
                if response.status_code == 200:
                    result = response.json()
                    full_response = result.get("response", "No response from the bot.")
                    
                    # Display raw query if enabled
                    if st.session_state.get("show_raw_query", False):
                        with st.expander("Raw Query"):
                            st.json(result.get("query_metadata", {}).get("es_query", {}))
                    # Display raw response if enabled
                    if st.session_state.get("show_raw_response", False):
                        with st.expander("Raw Response"):
                            st.json(result.get("query_metadata", {}).get("es_response", {}))
                else:
                    full_response = f"Error: {response.status_code} - {response.text}"
            except Exception as e:
                full_response = f"Error: {e}"
            
            message_placeholder.markdown(full_response)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})