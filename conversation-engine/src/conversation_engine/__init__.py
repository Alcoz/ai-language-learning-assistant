from typing import Literal

from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from conversation_engine.api_connection.ollama_connection import (
    generate_answer,
    get_instructions_prompt,
)
from conversation_engine.speech.stt import transcribe
from conversation_engine.speech.tts import synthesize

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
                model="qwen3.5:4b",
                messages=messages,
                session_id=query_content.session_id,
                prompt=instructions_prompt,
            ),
            media_type="text/plain",
        )

    @app.post("/stt")
    def speech_to_text(audio: UploadFile):
        return {"text": transcribe(audio.file, language=WHISPER_LANGUAGE)}

    @app.post("/tts")
    def text_to_speech(speech: Speech):
        if not speech.text.strip():
            raise HTTPException(status_code=422, detail="Text is empty")

        return Response(
            synthesize(speech.text, lang_code=KOKORO_LANG_CODE, voice=KOKORO_VOICE),
            media_type="audio/wav",
        )

    return app
