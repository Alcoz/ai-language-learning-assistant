import uuid
from collections.abc import Iterator

import requests
import streamlit as st


def init_session() -> None:
    """Create the message history and session id on first load."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "session_id" not in st.session_state:
        st.session_state.session_id = uuid.uuid4().hex


def reset_conversation() -> None:
    """Empty the history and start a new session."""
    st.session_state.messages = []
    st.session_state.session_id = uuid.uuid4().hex


def add_message(role: str, content: str, audio: bytes | None = None) -> None:
    """Append a message to the history, with its audio if any."""
    st.session_state.messages.append({"role": role, "content": content, "audio": audio})


def to_chat_messages(messages: list[dict]) -> list[dict]:
    """Keep only the role and content expected by the engine."""
    return [
        {"role": message["role"], "content": message["content"]} for message in messages
    ]


def stream_answer(
    back_url: str, messages: list[dict], session_id: str
) -> Iterator[str]:
    """Send the history to the engine and yield the teacher's answer as it streams."""
    with requests.post(
        f"{back_url.rstrip('/')}/query",
        json={"messages": to_chat_messages(messages), "session_id": session_id},
        stream=True,
        timeout=(5, 300),
    ) as response:
        response.raise_for_status()
        response.encoding = "utf-8"
        for chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if chunk:
                yield chunk
