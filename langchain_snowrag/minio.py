### `langchain_snowrag/minio.py`
### Embedding class for Snowflake
### Open-Source, hosted on https://github.com/DrBenjamin/RAG-on-Snow
### Please reach out to ben@seriousbenentertainment.org for any questions
import os
import logging
from minio import Minio
from minio.error import S3Error
import streamlit as st

logger = logging.getLogger(__name__)

# Setting up MinIO client from Streamlit secrets
def get_minio_client():
    """Creating and returning a MinIO client using Streamlit secrets."""
    return Minio(
        endpoint=st.secrets["MinIO"]["endpoint"].replace("http://", "").replace("https://", ""),
        access_key=st.secrets["MinIO"]["access_key"],
        secret_key=st.secrets["MinIO"]["secret_key"],
        secure=True
    )

# Uploading a file to MinIO and returning the public URL
def upload_file_to_minio(local_path, bucket=None):
    """
    Uploads a file to MinIO and returns the public URL.
    Args:
        local_path (str): Path to the local file to upload.
        bucket (str): MinIO bucket name. If None, uses secrets.
    Returns:
        str: Public URL to the uploaded file.
    """
    if bucket is None:
        bucket = st.secrets["MinIO"]["bucket"]
    minio_client = get_minio_client()
    filename = os.path.basename(local_path)
    try:
        # Creating the bucket if it does not exist
        found = minio_client.bucket_exists(bucket)
        if not found:
            minio_client.make_bucket(bucket)

        # Uploading the file
        minio_client.fput_object(bucket, filename, local_path)

        # Constructing the public URL
        url = f"{st.secrets['MinIO']['endpoint']}/{bucket}/{filename}"
        logger.info(f"Uploaded {local_path} to MinIO: {url}")
        return url
    except S3Error as err:
        logger.error(f"MinIO upload failed: {err}")
        raise
