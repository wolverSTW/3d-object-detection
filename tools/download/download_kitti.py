import os
import sys
import zipfile
import requests
from tqdm import tqdm

# KITTI Direct Official URLs
KITTI_URLS = {
    "images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "calib": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip",
    "velodyne": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
}

TARGET_DIR = "data/KITTI/raw"

def download_file(url: str, dest_path: str):
    """Downloads a file with progress bar."""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1MB
    
    print(f"Downloading: {url} -> {dest_path}")
    with open(dest_path, 'wb') as file, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=os.path.basename(dest_path)
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

def extract_zip(zip_path: str, extract_to: str):
    """Extracts a zip archive."""
    print(f"Extracting: {zip_path} to {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"Extracted: {zip_path}")

def main():
    os.makedirs(TARGET_DIR, exist_ok=True)
    
    # Download and Extract each component
    for key, url in KITTI_URLS.items():
        zip_name = os.path.basename(url)
        zip_dest = os.path.join(TARGET_DIR, zip_name)
        
        # Download if not exists
        if not os.path.exists(zip_dest):
            print(f"\n--- Downloading KITTI {key.upper()} ---")
            download_file(url, zip_dest)
        else:
            print(f"\nArchive already exists: {zip_dest}")
            
        # Extract
        print(f"--- Extracting KITTI {key.upper()} ---")
        extract_zip(zip_dest, TARGET_DIR)
        
    print("\n[SUCCESS] KITTI Dataset Download & Extraction Complete!")

if __name__ == "__main__":
    main()