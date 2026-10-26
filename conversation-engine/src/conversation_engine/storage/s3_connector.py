"""Interface with boto3 for any S3-compatible storage (MinIO, AWS S3, GCS)."""

import boto3
from botocore.config import Config


def connect_to_s3(
    endpoint_url: str,
    access_key_id: str,
    secret_access_key: str,
    region_name: str,
):
    """Create an S3 client, using the endpoint URL to target MinIO, AWS or GCS."""
    return boto3.client(
        "s3",
        endpoint_url=endpoint_url,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
        region_name=region_name,
        # MinIO and GCS reject the checksums that recent boto3 versions send by default
        config=Config(
            request_checksum_calculation="when_required",
            response_checksum_validation="when_required",
            connect_timeout=5,
            retries={"max_attempts": 2},
        ),
    )


def get_object_from_s3(s3_client, bucket: str, filepath: str) -> bytes:
    """Return the content of an object, failing if it is empty."""
    body = s3_client.get_object(Bucket=bucket, Key=filepath)["Body"].read()

    if not body:
        raise ValueError(f"{filepath} in {bucket} is empty")

    return body


def send_object_to_s3(
    s3_client,
    bucket: str,
    object: bytes | str,
    s3_filepath: str,
    content_type: str = "application/octet-stream",
) -> None:
    """Upload the content under the given key of the bucket."""
    s3_client.put_object(
        Bucket=bucket, Body=object, Key=s3_filepath, ContentType=content_type
    )


def get_folder_content_from_s3(s3_client, bucket_name: str, folder: str) -> list[str]:
    """List the names of the objects stored under a prefix of the bucket."""
    # list_objects_v2 returns at most 1000 keys per call: paginate to get them all
    paginator = s3_client.get_paginator("list_objects_v2")

    return [
        file["Key"].split("/")[-1]
        for page in paginator.paginate(Bucket=bucket_name, Prefix=folder)
        for file in page.get("Contents", [])
    ]
