import requests
from streamlit.runtime.uploaded_file_manager import UploadedFile


def speech_to_text(back_url: str, audio: UploadedFile) -> str:
    """Send the recorded audio to the engine and return its transcription."""
    response = requests.post(
        f"{back_url.rstrip('/')}/stt",
        files={"audio": (audio.name, audio.getvalue(), audio.type)},
        timeout=(5, 120),
    )
    response.raise_for_status()

    return response.json()["text"]


def text_to_speech(back_url: str, text: str) -> bytes:
    """Ask the engine to read the text aloud and return the WAV audio."""
    response = requests.post(
        f"{back_url.rstrip('/')}/tts",
        json={"text": text},
        timeout=(5, 120),
    )
    response.raise_for_status()

    return response.content
