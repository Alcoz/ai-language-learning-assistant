import os
import uuid
from collections.abc import Iterator

import requests
import streamlit as st
from streamlit.runtime.uploaded_file_manager import UploadedFile

DEFAULT_BACK_URL = os.environ.get("CONVERSATION_ENGINE_URL", "http://localhost:8000")
ROLE_LABELS = {"user": "Student", "assistant": "Teacher"}


def format_history(messages: list[dict]) -> str:
    # The API expects the whole conversation as a single string
    return "\n".join(
        f"{ROLE_LABELS[message['role']]}: {message['content']}" for message in messages
    )


def stream_answer(back_url: str, messages: list[dict], session_id: str) -> Iterator[str]:
    with requests.post(
        f"{back_url.rstrip('/')}/query",
        json={"messages": format_history(messages), "session_id": session_id},
        stream=True,
        timeout=(5, 300),
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk


def speech_to_text(back_url: str, audio: UploadedFile) -> str:
    response = requests.post(
        f"{back_url.rstrip('/')}/stt",
        files={"audio": (audio.name, audio.getvalue(), audio.type)},
        timeout=(5, 120),
    )
    response.raise_for_status()

    return response.json()["text"]


def text_to_speech(back_url: str, text: str) -> bytes:
    response = requests.post(
        f"{back_url.rstrip('/')}/tts",
        json={"text": text},
        timeout=(5, 120),
    )
    response.raise_for_status()

    return response.content


def ask_teacher(back_url: str) -> None:
    with st.chat_message("assistant"):
        try:
            answer = st.write_stream(stream_answer(back_url, st.session_state.messages, st.session_state.session_id))
        except requests.RequestException as error:
            st.error(f"Impossible de joindre le conversation-engine ({back_url}) : {error}")
            return

        audio = None
        try:
            with st.spinner("Génération de l'audio..."):
                audio = text_to_speech(back_url, answer)
            st.audio(audio, format="audio/wav", autoplay=True)
        except requests.RequestException as error:
            st.warning(f"Audio indisponible : {error}")

    st.session_state.messages.append({"role": "assistant", "content": answer, "audio": audio})


st.set_page_config(page_title="Language teacher", page_icon="🇮🇹")
st.title("🇮🇹 Language teacher")

with st.sidebar:
    back_url = st.text_input("URL du conversation-engine", value=DEFAULT_BACK_URL)
    if st.button("Nouvelle conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.session_id = uuid.uuid4().hex
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    ask_teacher(back_url)
elif not submission and not st.session_state.messages:
    # With an empty history the teacher opens the conversation by proposing topics
    ask_teacher(back_url)
