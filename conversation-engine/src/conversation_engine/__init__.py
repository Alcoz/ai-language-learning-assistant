import io
import os
from typing import Annotated, Literal

from fastapi import BackgroundTasks, FastAPI, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from conversation_engine.api_connection.ollama_connection import (
    generate_answer,
    get_instructions_prompt,
)
from conversation_engine.speech.stt import VERBATIM_PROMPT, transcribe
from conversation_engine.speech.tts import synthesize
from conversation_engine.storage.recordings import (
    archive_recording,
    get_recording_storage_error,
    is_recording_archive_enabled,
)

OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:4b")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "small")
# Codes used by Whisper (ISO 639-1) and Kokoro (lang_code + voice) for the taught language
WHISPER_LANGUAGE = "it"
KOKORO_LANG_CODE = "i"
KOKORO_VOICE = "if_sara"


class Message(BaseModel):
    """A single chat turn sent by the front."""

    role: Literal["user", "assistant"]
    content: str


class Query(BaseModel):
    """Body of a /query request: the conversation history and its Langfuse session."""

    messages: list[Message]
    session_id: str | None = None


class Speech(BaseModel):
    """Body of a /tts request: the text to read aloud."""

    text: str


def run_server():
    """Build the FastAPI app exposing the chat, speech-to-text and text-to-speech routes."""
    app = FastAPI()

    @app.post("/query")
    async def query(query_content: Query):
        instructions_prompt = get_instructions_prompt()
        system_prompt = instructions_prompt.compile(
            mother_tongue="french",
            language="italian",
            language_level="A1",
        )
        messages = [
            {"role": "system", "content": system_prompt},
            *(message.model_dump() for message in query_content.messages),
        ]

        return StreamingResponse(
            generate_answer(
                model=OLLAMA_MODEL,
                messages=messages,
                session_id=query_content.session_id,
                prompt=instructions_prompt,
            ),
            media_type="text/plain",
        )

    @app.post("/stt")
    def speech_to_text(
        audio: UploadFile,
        background_tasks: BackgroundTasks,
        session_id: Annotated[str | None, Form()] = None,
    ):
        audio_bytes = audio.file.read()
        text = transcribe(
            io.BytesIO(audio_bytes),
            language=WHISPER_LANGUAGE,
            model_size=WHISPER_MODEL_SIZE,
        )

        if is_recording_archive_enabled():
            background_tasks.add_task(
                archive_recording,
                audio=audio_bytes,
                filename=audio.filename,
                content_type=audio.content_type,
                transcription=text,
                stt_config={
                    "engine": "faster-whisper",
                    "model_size": WHISPER_MODEL_SIZE,
                    "language": WHISPER_LANGUAGE,
                    "initial_prompt": VERBATIM_PROMPT,
                },
                session_id=session_id,
            )

        return {"text": text}

    @app.get("/recordings/status")
    def recordings_status():
        if not is_recording_archive_enabled():
            return {"enabled": False, "error": None}

        return {"enabled": True, "error": get_recording_storage_error()}

    @app.post("/tts")
    def text_to_speech(speech: Speech):
        if not speech.text.strip():
            raise HTTPException(status_code=422, detail="Text is empty")

        return Response(
            synthesize(speech.text, lang_code=KOKORO_LANG_CODE, voice=KOKORO_VOICE),
            media_type="audio/wav",
        )

    return app
