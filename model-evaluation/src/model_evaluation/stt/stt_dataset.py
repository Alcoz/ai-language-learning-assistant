import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import PurePath

import pandas as pd
import soundfile
from conversation_engine.storage.s3_connector import (
    get_folder_content_from_s3,
    get_keys_by_suffix_from_s3,
    get_object_from_s3,
    send_object_to_s3,
)

LABEL_SUFFIX = ".label.json"
REFERENCE_FIELD = "reference"
DATASETS_PREFIX = "datasets/stt"


def extract_reference(label: dict) -> str | None:
    """Return the text written in the first annotation holding a reference."""
    for annotation in label["annotations"]:
        for result in annotation["result"]:
            if result["from_name"] == REFERENCE_FIELD and result["value"]["text"]:
                return result["value"]["text"][0]

    return None


def get_last_labeled_at(label: dict) -> str:
    """Return the date of the latest update of the annotations of a label."""
    return max(annotation["updated_at"] for annotation in label["annotations"])


def load_recordings(s3_client, bucket: str) -> tuple[pd.DataFrame, dict[str, bytes]]:
    """Build the dataset from the labels, and return the audio of each recording."""
    rows = []
    audios = {}

    for label_key in get_keys_by_suffix_from_s3(s3_client, bucket, LABEL_SUFFIX):
        label = json.loads(get_object_from_s3(s3_client, bucket, label_key))
        reference = extract_reference(label)
        if not reference:
            continue

        audio_key = label["audio_key"]
        audios[audio_key] = get_object_from_s3(s3_client, bucket, audio_key)
        rows.append(
            {
                "audio_key": audio_key,
                "reference": reference,
                "duration_seconds": soundfile.info(
                    io.BytesIO(audios[audio_key])
                ).duration,
                "labeled_at": get_last_labeled_at(label),
            }
        )

    rows.sort(key=lambda row: row["audio_key"])

    return pd.DataFrame(rows), audios


def compute_digest(dataset: pd.DataFrame) -> str:
    """Identify a dataset by its recordings and their references."""
    content = dataset[["audio_key", "reference"]].to_json(orient="records")
    return hashlib.sha256(content.encode()).hexdigest()[:12]


def get_dataset_key(digest: str) -> str:
    """Build the key of a published dataset."""
    return f"{DATASETS_PREFIX}/{digest}.json"


def describe_dataset(dataset: pd.DataFrame) -> dict:
    """Compute the metadata used to compare datasets."""
    sessions = dataset["audio_key"].map(lambda key: PurePath(key).parent.name)

    return {
        "digest": compute_digest(dataset),
        "created_at": datetime.now(UTC).isoformat(),
        "size": len(dataset),
        "num_sessions": sessions.nunique(),
        "total_duration_seconds": float(dataset["duration_seconds"].sum()),
        "last_labeled_at": dataset["labeled_at"].max(),
    }


def read_published_dataset(s3_client, bucket: str, digest: str) -> dict:
    """Read a dataset published in the bucket."""
    return json.loads(get_object_from_s3(s3_client, bucket, get_dataset_key(digest)))


def publish_dataset(s3_client, bucket: str, dataset: pd.DataFrame) -> dict:
    """Store the dataset once in the bucket, and return its metadata."""
    digest = compute_digest(dataset)
    published = get_folder_content_from_s3(s3_client, bucket, f"{DATASETS_PREFIX}/")

    if f"{digest}.json" in published:
        return read_published_dataset(s3_client, bucket, digest)["metadata"]

    metadata = describe_dataset(dataset)
    send_object_to_s3(
        s3_client,
        bucket,
        json.dumps(
            {"metadata": metadata, "recordings": dataset.to_dict("records")},
            ensure_ascii=False,
            indent=2,
        ),
        get_dataset_key(digest),
        "application/json",
    )

    return metadata


def produce_dataset(
    s3_client, bucket: str
) -> tuple[pd.DataFrame, dict[str, bytes], dict]:
    """Build a dataset from the current labels and publish it."""
    dataset, audios = load_recordings(s3_client, bucket)
    if dataset.empty:
        raise SystemExit("No labeled recording found: run `make export-labels` first")

    return dataset, audios, publish_dataset(s3_client, bucket, dataset)


def load_dataset(
    s3_client, bucket: str, digest: str
) -> tuple[pd.DataFrame, dict[str, bytes], dict]:
    """Reload a published dataset with the audio of its recordings."""
    published = read_published_dataset(s3_client, bucket, digest)
    dataset = pd.DataFrame(published["recordings"])
    audios = {
        audio_key: get_object_from_s3(s3_client, bucket, audio_key)
        for audio_key in dataset["audio_key"]
    }

    return dataset, audios, published["metadata"]
