# ai-language-learning-assistant

A conversational language teacher that runs entirely on your machine.
The student talks with the AI in the language they are learning, by typing or speaking.
The teacher matches the student's CEFR level and corrects mistakes one at a time, guiding the student toward the right answer rather than giving it straight away.

Current setup: a French-speaking student learning Italian at A1 level.

## Architecture

```
┌──────────────┐   HTTP    ┌─────────────────────┐
│    front     │ ────────► │ conversation-engine │
│ (Streamlit)  │ ◄──────── │      (FastAPI)      │
└──────────────┘  stream   └──────────┬──────────┘
                                      │
              ┌────────────┬──────────┼────────────┐
              ▼            ▼          ▼            ▼
           Ollama    faster-whisper  Kokoro     Langfuse
           (LLM)        (STT)        (TTS)     (tracing)
```

The repository is a `uv` workspace with two packages:

- **`conversation-engine`**: the API. It exposes three routes:
  - `/query`: sends the history to the LLM and streams back the answer;
  - `/stt`: transcribes an audio recording into text;
  - `/tts`: reads a text aloud and returns a WAV file.
- **`front`**: the chat interface. It keeps the conversation history and calls the API.

The teacher's instructions are in [instruction.txt](conversation-engine/src/conversation_engine/instruction.txt).

## Technologies and choices

| Component | Technology | Why |
|---|---|---|
| LLM | [Ollama](https://ollama.com) + `qwen3.5:4b` | Free and local, so conversations never go to a third party. A 4B model runs on a laptop, and thinking is disabled (`think=False`) so answers come without delay. |
| Prompting | Ollama chat API (`system` + messages) | The instructions are kept apart from the history: the model tells the rules from the conversation more reliably, and Langfuse traces stay readable. |
| Speech-to-text | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`small`, int8, CPU) | Good balance between quality and speed without a GPU. A prompt asks for a word-for-word transcription that keeps the student's mistakes: without them, the teacher would have nothing to correct. |
| Text-to-speech | [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) | Lightweight local model with a natural Italian voice (`if_sara`). |
| API | FastAPI + Uvicorn | Handles streaming and file uploads out of the box, and validates requests with Pydantic. |
| Interface | Streamlit | Provides chat, streaming, microphone recording and audio playback in a few lines. It is decoupled from the API, so it can be replaced. |
| Observability | [Langfuse](https://langfuse.com) | Traces each generation with its tokens and latency, and groups traces by session so a whole conversation can be replayed. |
| Packaging | [uv](https://docs.astral.sh/uv/) (workspace) | Two independent packages with a single lockfile. |
| Code quality | Ruff + pre-commit | Automatic linting and formatting on every commit. |

## Requirements

- Python 3.12 and [uv](https://docs.astral.sh/uv/getting-started/installation/)
- [Ollama](https://ollama.com/download) running, with the model pulled:

  ```bash
  ollama pull qwen3.5:4b
  ```

- A [Langfuse](https://langfuse.com/self-hosting) instance (self-hosted, at `http://localhost:3000` by default) and a pair of API keys

## Installation

```bash
uv sync
```

```bash
uv run pre-commit install
```

Create a `.env` file at the repository root:

```dotenv
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_BASE_URL=http://localhost:3000
```

## Running

From the repository root, start the API first, on port 8000:

```bash
uv run python conversation-engine/run_server.py
```

Then, in another terminal, start the interface:

```bash
uv run front
```

On first launch, the API downloads the Whisper `small` and Kokoro-82M models from Hugging Face.

The API URL defaults to `http://localhost:8000`. It can be changed with the `CONVERSATION_ENGINE_URL` environment variable or in the sidebar.

## Code quality

```bash
uv run pre-commit run --all-files
```
