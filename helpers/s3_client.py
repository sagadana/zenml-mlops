from __future__ import annotations

from functools import lru_cache
from typing import Annotated
from urllib.parse import urlparse

from zenml.client import Client

KEY_ACCESS_KEY_ID = "access_key_id"
KEY_SECRET_ACCESS_KEY = "secret_access_key"


@lru_cache(maxsize=16)
def _build_cached_client(
    seaweedfs_s3_internal_endpoint: str | None,
    seaweedfs_access_key_id: str | None,
    seaweedfs_secret_access_key: str | None,
):
    import boto3

    client_kwargs: dict[str, str] = {}
    if seaweedfs_s3_internal_endpoint:
        client_kwargs["endpoint_url"] = seaweedfs_s3_internal_endpoint
    if seaweedfs_access_key_id and seaweedfs_secret_access_key:
        client_kwargs["aws_access_key_id"] = seaweedfs_access_key_id
        client_kwargs["aws_secret_access_key"] = seaweedfs_secret_access_key
    return boto3.client("s3", **client_kwargs)


def get_s3_client(
    seaweedfs_s3_internal_endpoint: str | None = None,
    seaweedfs_access_key_id: str | None = None,
    seaweedfs_secret_access_key: str | None = None,
):
    """Build an S3 client for AWS or SeaweedFS endpoints."""
    return _build_cached_client(
        seaweedfs_s3_internal_endpoint,
        seaweedfs_access_key_id,
        seaweedfs_secret_access_key,
    )


def resolve_zenml_s3_credentials(
    zenml_local_s3_secret_name: str | None,
) -> tuple[
    Annotated[str | None, KEY_ACCESS_KEY_ID],
    Annotated[str | None, KEY_SECRET_ACCESS_KEY],
]:
    """Fetch SeaweedFS access key id and secret from ZenML secret store."""
    if not zenml_local_s3_secret_name:
        return None, None

    secret = Client().get_secret(zenml_local_s3_secret_name)
    return (
        secret.secret_values.get(KEY_ACCESS_KEY_ID),
        secret.secret_values.get(KEY_SECRET_ACCESS_KEY),
    )


def parse_s3_uri(s3_uri: str) -> tuple[str, str]:
    """Parse s3://bucket/key URI into (bucket, key)."""
    parsed = urlparse(s3_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key


def s3_get_object_text(
    s3_client,
    bucket: str,
    key: str,
    encoding: str = "utf-8",
) -> str:
    """Read an S3 object as decoded text."""
    return s3_client.get_object(Bucket=bucket, Key=key)["Body"].read().decode(encoding)


def s3_put_object_bytes(
    s3_client,
    bucket: str,
    key: str,
    body: bytes,
    content_type: str | None = None,
) -> None:
    """Write bytes to S3 object with optional content type."""
    put_kwargs: dict[str, object] = {
        "Bucket": bucket,
        "Key": key,
        "Body": body,
    }
    if content_type:
        put_kwargs["ContentType"] = content_type
    s3_client.put_object(**put_kwargs)
