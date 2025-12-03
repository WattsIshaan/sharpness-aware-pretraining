import os
import sys
import subprocess
from pathlib import Path

import yaml


CONFIG_PATH = "/home/iwatts/catastrophic-forgetting/configs/official-0425/OLMo2-1B-stage1.yaml"
LOCAL_TRAIN_DIR = "/data/user_data/iwatts/datasets/dclm/train"
GCS_TRAIN_DIR = "gs://cmu-gpucloud-jspringe/shared/datasets/OLMo/dclm/train/"
URL_PREFIX_TO_STRIP = "http://olmo-data.org/"


def read_config_urls(config_path: str) -> list:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    try:
        paths = cfg["data"]["paths"]
    except Exception as e:
        raise KeyError(f"Could not find data.paths in config: {e}")
    if not isinstance(paths, list):
        raise TypeError("data.paths must be a list of URLs")
    urls = [p for p in paths if isinstance(p, str) and p.startswith("http")]
    if not urls:
        raise ValueError("No HTTP URLs found in data.paths")
    return urls


def transform_filename(url: str) -> str:
    if url.startswith(URL_PREFIX_TO_STRIP):
        trimmed = url[len(URL_PREFIX_TO_STRIP):]
    else:
        trimmed = url
    trimmed = trimmed.replace("/", "_")
    # trimmed_gcs = trimmed[:-10] + "/" + trimmed[-9:]
    return trimmed


def ensure_local_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def gsutil_ls(path: str) -> bool:
    result = subprocess.run(["gsutil", "ls", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0


def ensure_gcs_dir(path: str) -> None:
    if not path.endswith("/"):
        raise ValueError("GCS directory path must end with '/'")
    _exists = gsutil_ls(path)
    # Directory-like prefixes in GCS are virtual; copy operations will create them if needed.
    # We only check existence to satisfy the requirement.


def download_via_wget(url: str, dest_path: str) -> None:
    # Use wget with retries and timeout; -O selects output path
    subprocess.check_call([
        "wget",
        "--tries=5",
        "-O",
        dest_path,
        url,
    ])


def upload_to_gcs(local_path: str, gcs_dir: str) -> None:
    subprocess.check_call(["gsutil", "-m", "cp", local_path, gcs_dir])


def main() -> int:
    urls = read_config_urls(CONFIG_PATH)
    ensure_local_dir(LOCAL_TRAIN_DIR)
    ensure_gcs_dir(GCS_TRAIN_DIR)

    for url in urls:
        filename = transform_filename(url)
        local_file = os.path.join(LOCAL_TRAIN_DIR, filename)

        print(f"Downloading: {url} -> {local_file}")
        download_via_wget(url, local_file)

        print(f"Uploading to GCS: {local_file} -> {GCS_TRAIN_DIR}")
        upload_to_gcs(local_file, GCS_TRAIN_DIR)

        os.remove(local_file)
        print(f"Deleted local file: {local_file}")
        

    print("All files downloaded and synced to GCS.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)

