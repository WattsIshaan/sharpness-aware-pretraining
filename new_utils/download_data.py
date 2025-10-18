import sys
import os
import yaml
import subprocess

def url_to_filename(url):
    # Replace s3://ai2-llm/ with https://olmo-data.org/
    if url.startswith("s3://ai2-llm/"):
        url = url.replace("s3://ai2-llm/", "https://olmo-data.org/")
    # Transform the path: remove the base URL and replace '/' with '_' for filename
    if url.startswith("https://olmo-data.org/"):
        path_part = url[len("https://olmo-data.org/"):]
        filename = path_part.replace("/", "_")
    else:
        filename = os.path.basename(url)
    return filename, url

def download_file(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    filename, url = url_to_filename(url)
    local_path = os.path.join(dest_dir, filename)
    print(f"Downloading {url} to {local_path}...")
    result = subprocess.run(['wget', '-O', local_path, url])
    if result.returncode != 0:
        print(f"Failed to download {url}")
        return None
    else:
        print(f"Downloaded {url}")
        return local_path

def count_files(dest_dir):
    if not os.path.exists(dest_dir):
        return 0
    return len([f for f in os.listdir(dest_dir) if os.path.isfile(os.path.join(dest_dir, f))])

def download_train_data(config, output_dir):
    # Download files listed in data:paths
    if 'data' not in config or 'paths' not in config['data']:
        print("YAML file must have a 'data' field with a 'paths' subfield (list of URLs).")
        sys.exit(1)
    data_files = config['data']['paths']
    if not isinstance(data_files, list):
        print("The 'paths' subfield under 'data' should be a list of URLs.")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)
    start_idx = count_files(output_dir)
    if start_idx > 0:
        print(f"Found {start_idx} files already in {output_dir}. Resuming from index {max(0, start_idx-1)} (redownloading last file and onwards).")
        start_idx = max(0, start_idx-1)
    for url in data_files[start_idx:]:
        download_file(url, output_dir)

def download_val_data(config, output_dir):
    # Download files listed in evaluators:...:datasets
    if 'evaluators' not in config or not isinstance(config['evaluators'], list):
        print("No 'evaluators' field found or not a list.")
        return

    os.makedirs(output_dir, exist_ok=True)
    url_list = []
    for evaluator in config['evaluators']:
        if (
            isinstance(evaluator, dict)
            and 'data' in evaluator
            and 'datasets' in evaluator['data']
        ):
            datasets = evaluator['data']['datasets']
            for key, url in datasets.items():
                urls = [url] if isinstance(url, str) else url
                if not isinstance(urls, list):
                    print(f"Unexpected type for datasets[{key}]: {type(url)}")
                    continue
                for u in urls:
                    url_list.append(u)

    start_idx = count_files(output_dir)
    if start_idx > 0:
        print(f"Found {start_idx} files already in {output_dir}. Resuming from index {max(0, start_idx-1)} (redownloading last file and onwards).")
        start_idx = max(0, start_idx-1)
    for url in url_list[start_idx:]:
        download_file(url, output_dir)

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python utils/download_data.py <config.yaml> <output_dir> <data_type>")
        print("data_type: train or val")
        sys.exit(1)
    yaml_path = sys.argv[1]
    output_dir = sys.argv[2]
    data_type = sys.argv[3].lower()
    with open(yaml_path, 'r') as f:
        config = yaml.safe_load(f)
    if data_type == "train":
        download_train_data(config, os.path.join(output_dir, "train"))
    elif data_type == "val":
        download_val_data(config, os.path.join(output_dir, "val"))
    elif data_type == "all":
        download_val_data(config, os.path.join(output_dir, "val"))
        download_train_data(config, os.path.join(output_dir, "train"))
    else:
        print("data_type must be 'train' or 'val'")
        sys.exit(1)
        download_train_data(config, os.path.join(output_dir, "train"))
