# Copyright 2022 MosaicML Composer authors
# SPDX-License-Identifier: Apache-2.0

"""Download handling for :class:`StreamingDataset`.
"""

import os
import shutil
import textwrap
import time
from urllib.parse import urlparse

__all__ = ["download_or_wait"]


def download_from_s3(remote: str, local: str, timeout: float) -> None:
    """Download a file from remote to local.
    Args:
        remote (str): Remote path (S3).
        local (str): Local path (local filesystem).
        timeout (float): How long to wait for shard to download before raising an exception.
    """
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise ImportError(
            textwrap.dedent("""\
            Composer was installed without streaming support. To use streaming with Composer, run: `pip install mosaicml
            [streaming]` if using pip or `conda install -c conda-forge monai` if using Anaconda""")) from e

    obj = urlparse(remote)
    if obj.scheme != 's3':
        raise ValueError(f"Expected obj.scheme to be 's3', got {obj.scheme} for remote={remote}")

    config = Config(read_timeout=timeout)
    s3 = boto3.client('s3', config=config)
    s3.download_file(obj.netloc, obj.path[1:], local)


def download_from_local(remote: str, local: str) -> None:
    """Download a file from remote to local.
    Args:
        remote (str): Remote path (local filesystem).
        local (str): Local path (local filesystem).
    """
    local_tmp = local + ".tmp"
    if os.path.exists(local_tmp):
        os.remove(local_tmp)
    shutil.copy(remote, local_tmp)
    os.rename(local_tmp, local)


def dispatch_download(remote, local, timeout: float):
    if os.path.exists(local):
        return

    local_dir = os.path.dirname(local)
    os.makedirs(local_dir, exist_ok=True)

    if remote.startswith('s3://'):
        download_from_s3(remote, local, timeout)
    else:
        download_from_local(remote, local)


def download_or_wait(remote: str, local: str, wait: bool = False, max_retries: int = 2, timeout: float = 60) -> None:
    """Downloads a file from remote to local, or waits for it to be downloaded.
       Does not do any thread safety checks, so that must be handled separately.
    Args:
        remote (str): Remote path (S3 or local filesystem).
        local (str): Local path (local filesystem).
        wait (bool, default False): Whether to wait (up to ``timeout`` seconds) for the file to arrive.
        max_retries (int, default 2): Number of download attempts before giving up.
        timeout (float, default 60): How long to wait for shard to download before raising an exception.
    """
    ok = False
    error_msgs = []
    for _ in range(1 + max_retries):
        try:
            if wait:
                start = time.time()
                while not os.path.exists(local):
                    if time.time() - start > timeout:
                        raise TimeoutError(f"Waited longer than {timeout}s for other worker to download {local}.")
                    time.sleep(0.25)
            else:
                dispatch_download(remote, local, timeout=timeout)
            ok = True
            break
        except Exception as e:  # Retry for all causes of failure.
            error_msgs.append(str(e))
            continue
    if not ok:
        raise RuntimeError(f"Failed to download {remote} -> {local}. Got error messages:\n{error_msgs}")
