import sys
import os
import yaml
import subprocess

def download_file(url, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    filename = os.path.basename(url)
    local_path = os.path.join(dest_dir, filename)
    print(f"Downloading {url} to {local_path}...")
    result = subprocess.run(['wget', '-O', local_path, url])
    if result.returncode != 0:
        print(f"Failed to download {url}")
        return None
    else:
        print(f"Downloaded {url}")
        return local_path

def download_train_data(config, output_dir):
    # Download files listed in data:paths
    if 'data' not in config or 'paths' not in config['data']:
        print("YAML file must have a 'data' field with a 'paths' subfield (list of URLs).")
        sys.exit(1)
    data_files = config['data']['paths']
    if not isinstance(data_files, list):
        print("The 'paths' subfield under 'data' should be a list of URLs.")
        sys.exit(1)
    for url in data_files:
        download_file(url, output_dir)

def download_val_data(config, output_dir):
    # Download files listed in evaluators:...:datasets
    if 'evaluators' not in config or not isinstance(config['evaluators'], list):
        print("No 'evaluators' field found or not a list.")
        return
    for evaluator in config['evaluators']:
        if (
            isinstance(evaluator, dict)
            and 'data' in evaluator
            and 'datasets' in evaluator['data']
        ):
            datasets = evaluator['data']['datasets']
            for key, url in datasets.items():
                # url can be a list or a single string
                urls = [url] if isinstance(url, str) else url
                if not isinstance(urls, list):
                    print(f"Unexpected type for datasets[{key}]: {type(url)}")
                    continue
                for u in urls:
                    download_file(u, output_dir)

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
        download_train_data(config, output_dir)
    elif data_type == "val":
        download_val_data(config, output_dir)
    else:
        print("data_type must be 'train' or 'val'")
        sys.exit(1)
