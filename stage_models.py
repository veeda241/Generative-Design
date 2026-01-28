import os
import requests
import shutil

def stage():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    old_cache = os.path.join(base_dir, 'backend', 'point_e_model_cache')
    new_cache = os.path.join(base_dir, 'backend', 'point_e_cache_fixed')
    
    print(f"Creating {new_cache}...")
    os.makedirs(new_cache, exist_ok=True)
    
    # Copy existing files if they exist and are not locked
    for model in ['base_40m.pt', 'ViT-L-14.pt']:
        src = os.path.join(old_cache, model)
        dst = os.path.join(new_cache, model)
        if os.path.exists(src) and not os.path.exists(dst):
            print(f"Copying {model}...")
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Failed to copy {model}: {e}")

    # Download upsample model
    upsample_url = "https://openaipublic.azureedge.net/main/point-e/upsample_40m.pt"
    upsample_dst = os.path.join(new_cache, 'upsample_40m.pt')
    
    if not os.path.exists(upsample_dst):
        print(f"Downloading upsample model to {upsample_dst}...")
        try:
            response = requests.get(upsample_url, stream=True, timeout=60)
            response.raise_for_status()
            with open(upsample_dst, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download successful.")
        except Exception as e:
            print(f"Download failed: {e}")
    else:
        print("Upsample model already exists in fixed cache.")

if __name__ == "__main__":
    stage()
