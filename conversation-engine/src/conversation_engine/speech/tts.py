import io
import re
from functools import cache

import numpy as np
import soundfile as sf
from kokoro import KPipeline

SAMPLE_RATE = 24000
# Markdown symbols the teacher may use, which should not be read aloud
MARKDOWN_SYMBOLS = re.compile(r"[*_#`>~]")


@cache
def get_pipeline(lang_code: str) -> KPipeline:
    return KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")


def synthesize(
    text: str,
    lang_code: str,
    voice: str,
) -> bytes:
    text = MARKDOWN_SYMBOLS.sub("", text)
    chunks = [audio for _, _, audio in get_pipeline(lang_code)(text, voice=voice)]

    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(chunks), SAMPLE_RATE, format="WAV")

    return buffer.getvalue()
