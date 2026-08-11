import streamlit as st
import requests

BACK_URL = "http://0.0.0.0:8000/query"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Buongiorno, come stai ?"}]

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Come stai ?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        response = requests.post(
            BACK_URL, 
            json={'messages': prompt},
            stream=True
        )

        for chunk in response.iter_content(
            chunk_size=None,
            decode_unicode=True,
        ):
            if chunk:
                full_response += chunk
                message_placeholder.markdown(full_response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
