import argparse
import io
import os
import time

import jiwer
import mlflow
import pandas as pd
from conversation_engine import WHISPER_LANGUAGE
from conversation_engine.speech.stt import VERBATIM_PROMPT, get_model, transcribe
from conversation_engine.storage.recordings import get_s3_client
from dotenv import load_dotenv

from model_evaluation.stt.stt_dataset import (
    get_dataset_key,
    load_dataset,
    produce_dataset,
)

EXPERIMENT_NAME = "STT Experiment"
DATASET_NAME = "stt-labeled-recordings"
WHISPER_MODELS = ["tiny", "base", "small", "medium", "large-v3-turbo"]

normalize_text = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemovePunctuation(),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
    ]
)


def transcribe_dataset(
    model: str,
    dataset: pd.DataFrame,
    audios: dict[str, bytes],
) -> pd.DataFrame:
    """Transcribe every recording and add the text and latency to the dataset."""
    get_model(model)
    transcriptions = []
    latencies = []

    for audio_key in dataset["audio_key"]:
        started_at = time.perf_counter()
        transcriptions.append(
            transcribe(
                io.BytesIO(audios[audio_key]),
                language=WHISPER_LANGUAGE,
                model_size=model,
            )
        )
        latencies.append(time.perf_counter() - started_at)

    return dataset.assign(transcription=transcriptions, latency_seconds=latencies)


def compute_metrics(results: pd.DataFrame) -> dict[str, float]:
    """Compute the error rates and the speed over the whole dataset."""
    references = [normalize_text(text) for text in results["reference"]]
    transcriptions = [normalize_text(text) for text in results["transcription"]]

    return {
        "wer": jiwer.wer(references, transcriptions),
        "cer": jiwer.cer(references, transcriptions),
        "rtf": results["latency_seconds"].sum() / results["duration_seconds"].sum(),
        "latency_seconds_mean": results["latency_seconds"].mean(),
    }


def main() -> None:
    """Evaluate each Whisper model on the labeled recordings."""
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="digest of a published dataset to reuse")
    digest = parser.parse_args().dataset

    bucket = os.environ["BUCKET_NAME"]
    if digest:
        dataset, audios, metadata = load_dataset(get_s3_client(), bucket, digest)
    else:
        dataset, audios, metadata = produce_dataset(get_s3_client(), bucket)

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow_dataset = mlflow.data.from_pandas(
        dataset,
        source=f"s3://{bucket}/{get_dataset_key(metadata['digest'])}",
        name=DATASET_NAME,
        digest=metadata["digest"],
    )

    for model in WHISPER_MODELS:
        with mlflow.start_run(run_name=model):
            mlflow.log_input(mlflow_dataset, context="evaluation")
            mlflow.log_params(
                {
                    "engine": "faster-whisper",
                    "model": model,
                    "language": WHISPER_LANGUAGE,
                    "initial_prompt": VERBATIM_PROMPT,
                    **{f"dataset_{name}": value for name, value in metadata.items()},
                }
            )

            results = transcribe_dataset(model, dataset, audios)
            metrics = compute_metrics(results)
            mlflow.log_metrics(metrics)
            mlflow.log_table(results, "transcriptions.json")

            print(f"{model}: {metrics}")


if __name__ == "__main__":
    main()
