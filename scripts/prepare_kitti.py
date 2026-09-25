import os
import zipfile
import urllib.request
import yaml
from tqdm import tqdm

class TqdmUpTo(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def download_file(url, output_path):
    print(f"Downloading {url} to {output_path}...")
    with TqdmUpTo(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=os.path.basename(output_path)) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

def extract_zip(zip_path, extract_to):
    print(f"Extracting {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    os.remove(zip_path)
    print(f"Extracted and removed {zip_path}")

def main():
    config = load_config("configs/dataset/kitti.yaml")
    base_url = config['dataset']['download_url']
    target_dir = config['dataset']['root_dir']
    
    os.makedirs(target_dir, exist_ok=True)
    
    # KITTI 3D Detection required files
    files_to_download = {
        "data_object_image_2.zip": f"{base_url}/data_object_image_2.zip",
        "data_object_label_2.zip": f"{base_url}/data_object_label_2.zip",
        "data_object_calib.zip": f"{base_url}/data_object_calib.zip"
    }

    for file_name, url in files_to_download.items():
        file_path = os.path.join(target_dir, file_name)
        if not os.path.exists(file_path):
            try:
                download_file(url, file_path)
                extract_zip(file_path, target_dir)
            except Exception as e:
                print(f"Failed to download or extract {file_name}: {e}")
        else:
            print(f"File {file_name} already exists. Skipping.")

if __name__ == "__main__":
    main()
