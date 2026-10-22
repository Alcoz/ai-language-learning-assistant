import os

import requests
import streamlit as st

from front.ai_calls import speech_to_text, text_to_speech
from front.utils import add_message, init_session, reset_conversation, stream_answer

DEFAULT_BACK_URL = os.environ.get("CONVERSATION_ENGINE_URL", "http://localhost:8000")


def ask_teacher(back_url: str) -> None:
    """Display the teacher's streamed answer, play its audio and save it to the history."""
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(
                stream_answer(
                    back_url, st.session_state.messages, st.session_state.session_id
                )
            )
        except requests.RequestException as error:
            st.error(
                f"Impossible de joindre le conversation-engine ({back_url}) : {error}"
            )
            return

        audio = None
        try:
            with st.spinner("Génération de l'audio..."):
                audio = text_to_speech(back_url, answer)
            st.audio(audio, format="audio/wav", autoplay=True)
        except requests.RequestException as error:
            st.warning(f"Audio indisponible : {error}")

    add_message("assistant", answer, audio)


st.set_page_config(page_title="Language teacher", page_icon="🇮🇹")
st.title("🇮🇹 Language teacher")

with st.sidebar:
    back_url = st.text_input("URL du conversation-engine", value=DEFAULT_BACK_URL)
    if st.button("Nouvelle conversation", use_container_width=True):
        reset_conversation()
        st.rerun()

init_session()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("audio"):
            st.audio(message["audio"], format="audio/wav")

submission = st.chat_input("Scrivi o registra il tuo messaggio...", accept_audio=True)

prompt = None
if submission and submission.audio:
    try:
        with st.spinner("Transcription..."):
            prompt = speech_to_text(back_url, submission.audio)
    except requests.RequestException as error:
        st.error(f"Transcription impossible : {error}")
elif submission:
    prompt = submission.text

if prompt:
    add_message("user", prompt)
    with st.chat_message("user"):
        st.markdown(prompt)

    ask_teacher(back_url)
elif not submission and not st.session_state.messages:
    # With an empty history the teacher opens the conversation by proposing topics
    ask_teacher(back_url)
