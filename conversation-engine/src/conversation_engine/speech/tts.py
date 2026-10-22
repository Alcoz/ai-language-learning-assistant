import io
import re
import warnings
from functools import cache

import numpy as np
import soundfile as sf
from kokoro import KPipeline

# Harmless warnings emitted by Kokoro's model code when the pipeline is built
warnings.filterwarnings(
    "ignore", message=r".*weight_norm.*is deprecated", category=FutureWarning
)
warnings.filterwarnings(
    "ignore",
    message="dropout option adds dropout after all but last recurrent layer",
    category=UserWarning,
)

SAMPLE_RATE = 24000
# Markdown symbols the teacher may use, which should not be read aloud
MARKDOWN_SYMBOLS = re.compile(r"[*_#`>~]")


@cache
def get_pipeline(lang_code: str) -> KPipeline:
    """Load the Kokoro pipeline once per language and reuse it."""
    return KPipeline(lang_code=lang_code, repo_id="hexgrad/Kokoro-82M")


def synthesize(
    text: str,
    lang_code: str,
    voice: str,
) -> bytes:
    """Convert text to WAV audio, without reading Markdown symbols aloud."""
    text = MARKDOWN_SYMBOLS.sub("", text)
    chunks = [audio for _, _, audio in get_pipeline(lang_code)(text, voice=voice)]

    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(chunks), SAMPLE_RATE, format="WAV")

    return buffer.getvalue()
