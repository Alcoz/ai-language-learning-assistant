import json
import logging
import os
import uuid
from datetime import UTC, datetime
from functools import cache
from pathlib import PurePath

from conversation_engine.storage.s3_connector import connect_to_s3, send_object_to_s3

logger = logging.getLogger(__name__)

RAW_PREFIX = "raw"


def is_recording_archive_enabled() -> bool:
    """Tell whether the student's recordings should be kept for the STT benchmark."""
    return os.environ.get("ARCHIVE_RECORDINGS", "false").lower() == "true"


@cache
def get_s3_client():
    """Connect once to the storage configured in the environment."""
    return connect_to_s3(
        endpoint_url=os.environ["S3_API"],
        access_key_id=os.environ["S3_ACCESS_KEY"],
        secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        region_name=os.environ["REGION_NAME"],
    )


def get_recording_storage_error() -> str | None:
    """Check that the bucket is reachable, and explain why when it is not."""
    try:
        get_s3_client().head_bucket(Bucket=os.environ["BUCKET_NAME"])
    except KeyError as error:
        return f"Missing environment variable {error}"
    except Exception as error:
        return str(error)

    return None


def build_recording_key(session_id: str | None, recorded_at: datetime) -> str:
    """Build the key, without extension, grouping recordings by day and session."""
    return (
        f"{RAW_PREFIX}/{recorded_at:%Y-%m-%d}/{session_id or 'no-session'}"
        f"/{uuid.uuid4()}"
    )


def archive_recording(
    audio: bytes,
    filename: str | None,
    content_type: str | None,
    transcription: str,
    stt_config: dict,
    session_id: str | None,
) -> None:
    """Store the raw audio and its metadata, without failing the student's request."""
    recorded_at = datetime.now(UTC)
    key = build_recording_key(session_id, recorded_at)
    extension = PurePath(filename or "").suffix or ".wav"

    try:
        bucket = os.environ["BUCKET_NAME"]
        metadata = {
            "audio": f"s3://{bucket}/{key}{extension}",
            "audio_key": f"{key}{extension}",
            "recorded_at": recorded_at.isoformat(),
            "session_id": session_id,
            "original_filename": filename,
            "content_type": content_type,
            "transcription": transcription,
            "stt": stt_config,
        }
        send_object_to_s3(
            get_s3_client(),
            bucket,
            audio,
            metadata["audio_key"],
            content_type or "audio/wav",
        )
        send_object_to_s3(
            get_s3_client(),
            bucket,
            json.dumps(metadata, ensure_ascii=False, indent=2),
            f"{key}.json",
            "application/json",
        )
    except Exception:
        logger.exception("Could not archive the recording %s", key)
