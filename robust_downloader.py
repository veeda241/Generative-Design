import requests
import os
from tqdm import tqdm

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 # 1 Kibibyte
    
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    with open(filename, 'wb') as file, tqdm(
        desc=os.path.basename(filename),
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in response.iter_content(block_size):
            size = file.write(data)
            bar.update(size)

if __name__ == "__main__":
    url = "https://openaipublic.azureedge.net/main/point-e/upsample_40m.pt"
    dest = "backend/point_e_model_cache/upsample_40m.pt"
    
    # Pre-clean
    if os.path.exists(dest + ".lock"): os.remove(dest + ".lock")
    if os.path.exists(dest + ".tmp"): os.remove(dest + ".tmp")
    
    try:
        download_file(url, dest)
        print("Successfully downloaded upsample_40m.pt")
    except Exception as e:
        print(f"Download failed: {e}")
