import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile


def speech_to_text(back_url: str, audio: UploadedFile, session_id: str) -> str:
    """Send the recorded audio to the engine and return its transcription."""
    response = requests.post(
        f"{back_url.rstrip('/')}/stt",
        files={"audio": (audio.name, audio.getvalue(), audio.type)},
        data={"session_id": session_id},
        timeout=(5, 120),
    )
    response.raise_for_status()

    return response.json()["text"]


def get_recording_storage_error(back_url: str) -> str | None:
    """Ask the engine whether recordings are archived but their storage is unreachable."""
    response = requests.get(
        f"{back_url.rstrip('/')}/recordings/status", timeout=(5, 30)
    )
    response.raise_for_status()

    return response.json()["error"]


def text_to_speech(back_url: str, text: str) -> bytes:
    """Ask the engine to read the text aloud and return the WAV audio."""
    response = requests.post(
        f"{back_url.rstrip('/')}/tts",
        json={"text": text},
        timeout=(5, 120),
    )
    response.raise_for_status()

    return response.content
