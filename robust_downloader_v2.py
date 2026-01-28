import requests
import os
import time

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    for attempt in range(5):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            
            with open(filename, 'wb') as file:
                downloaded = 0
                for data in response.iter_content(chunk_size=8192):
                    downloaded += len(data)
                    file.write(data)
                    if downloaded % (1024 * 1024) == 0: # Print every MB
                        print(f"Downloaded {downloaded / (1024*1024):.1f}MB / {total_size / (1024*1024):.1f}MB")
            
            print(f"Successfully downloaded {filename}")
            return True
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(5)
    return False

if __name__ == "__main__":
    url = "https://openaipublic.azureedge.net/main/point-e/upsample_40m.pt"
    dest = "backend/point_e_model_cache/upsample_40m_final.pt"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if download_file(url, dest):
        print("Download complete.")
    else:
        print("Download failed after multiple attempts.")
