import boto3
from pathlib import Path

def upload_folder_to_s3(local_folder: Path, bucket: str, s3_prefix: str):
    """
    Upload all files in local_folder recursively to S3 under s3_prefix.
    """
    s3 = boto3.client("s3")
    for file_path in local_folder.rglob("*"):
        if file_path.is_file():
            relative_path = file_path.relative_to(local_folder)
            s3_key = f"{s3_prefix}/{relative_path.as_posix()}"
            print(f"[UPLOAD] {file_path} -> s3://{bucket}/{s3_key}")
            s3.upload_file(str(file_path), bucket, s3_key)

def download_s3_folder(bucket, prefix, local_path: Path, s3):
    """
    Recursively download all files from an S3 folder (prefix) to a local path.
    Skips files that already exist locally.

    Args:
        bucket (str): Name of the S3 bucket.
        prefix (str): The S3 folder prefix to download.
        local_path (Path): Local folder where files will be stored.
        s3 (boto3.client): Boto3 S3 client instance.
    """
    # Ensure the base local directory exists
    local_path.mkdir(parents=True, exist_ok=True)

    paginator = s3.get_paginator("list_objects_v2")
    downloaded_count = 0
    skipped_count = 0

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if "Contents" not in page:
            continue

        for obj in page.get("Contents", []):
            key = obj["Key"]

            # Skip pseudo-directories
            if key.endswith("/"):
                continue

            # Create relative path inside local_path
            relative_path = Path("/".join(key.split("/")[1:]))
            file_path = local_path / relative_path

            # Skip if file already exists
            if file_path.exists():
                skipped_count += 1
                continue

            # Ensure parent directories exist
            file_path.parent.mkdir(parents=True, exist_ok=True)

            # Download the file
            s3.download_file(bucket, key, str(file_path))
            downloaded_count += 1

    print(
        f"[INFO] S3 download completed. Downloaded {downloaded_count} files, skipped {skipped_count} existing files."
    )
