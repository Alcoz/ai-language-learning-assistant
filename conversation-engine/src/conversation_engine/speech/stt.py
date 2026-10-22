from functools import cache
from typing import BinaryIO

from faster_whisper import WhisperModel

# Asks Whisper to keep the student's mistakes instead of "fixing" them
VERBATIM_PROMPT = (
    "Trascrizione letterale, con tutti gli errori di grammatica e pronuncia."
)


@cache
def get_model(model_size: str) -> WhisperModel:
    """Load the Whisper model once per size and reuse it."""
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe(
    audio: BinaryIO,
    language: str,
    model_size: str = "small",
) -> str:
    """Transcribe the audio word for word, keeping the student's mistakes."""
    segments, _ = get_model(model_size).transcribe(
        audio,
        language=language,
        initial_prompt=VERBATIM_PROMPT,
        vad_filter=True,
    )

    return " ".join(segment.text.strip() for segment in segments)
