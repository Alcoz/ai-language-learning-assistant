from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from conversation_engine.api_connection.ollama_connection import generate_answer
from conversation_engine.speech.stt import transcribe
from conversation_engine.speech.tts import synthesize

# Codes used by Whisper (ISO 639-1) and Kokoro (lang_code + voice) for the taught language
WHISPER_LANGUAGE = "it"
KOKORO_LANG_CODE = "i"
KOKORO_VOICE = "if_sara"

class Query(BaseModel):
    messages: str

class Speech(BaseModel):
    text: str

def run_server():
    app = FastAPI()

    with open("conversation-engine/src/conversation_engine/instruction.txt") as instruction_file:
        instructions = instruction_file.read()

    @app.post("/query")
    async def query(query_content: Query):

        prompt = instructions.format(
            mother_tongue="french",
            language="italian",
            language_level="A1",
            history=query_content.messages
        )

        print(prompt)

        return StreamingResponse(
            generate_answer(
                model="qwen3.5:4b",
                messages=prompt
            ),
            media_type="text/plain",
        )

    # Sync endpoints: FastAPI runs them in a threadpool, so the models don't block the event loop
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
