import json
import os
from pathlib import PurePath

from label_studio_sdk.client import LabelStudio

from conversation_engine.storage.recordings import get_s3_client
from conversation_engine.storage.s3_connector import send_object_to_s3


def build_label_key(audio_key: str) -> str:
    """Build the key of the label, in the same folder as its audio."""
    return str(PurePath(audio_key).with_suffix(".label.json"))


def get_submitted_annotations(task) -> list[dict]:
    """Keep the annotations of a task that were not skipped in Label Studio."""
    return [
        annotation
        for annotation in task.annotations
        if not annotation.get("was_cancelled")
    ]


def export_labels() -> int:
    """Write the labels of each annotated task next to its audio, and count them."""
    client = LabelStudio(
        base_url=os.environ["LABEL_STUDIO_URL"],
        api_key=os.environ["LABEL_STUDIO_API_KEY"],
    )
    tasks = client.tasks.list(
        project=int(os.environ["LABEL_STUDIO_PROJECT_ID"]),
        fields="all",
        only_annotated=True,
    )

    exported = 0
    for task in tasks:
        annotations = get_submitted_annotations(task)
        if not annotations:
            continue

        audio_key = task.data["audio_key"]
        label = {
            "task_id": task.id,
            "audio_key": audio_key,
            "session_id": task.data["session_id"],
            "transcription": task.data["transcription"],
            "annotations": annotations,
        }
        send_object_to_s3(
            get_s3_client(),
            os.environ["BUCKET_NAME"],
            json.dumps(label, ensure_ascii=False, indent=2),
            build_label_key(audio_key),
            "application/json",
        )
        exported += 1

    return exported
