import os
import requests

def download_dataset():
    url = "https://raw.githubusercontent.com/MainakRepositor/Datasets/master/rainfall%20in%20india%201901-2015.csv"
    dest_dir = "data"
    dest_path = os.path.join(dest_dir, "rainfall_in_india_1901-2015.csv")
    
    # Create data directory if it doesn't exist
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        print(f"Created directory: {dest_dir}")
        
    print(f"Downloading dataset from {url}...")
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        
        with open(dest_path, 'wb') as f:
            f.write(response.content)
            
        print(f"Successfully downloaded and saved dataset to {dest_path}")
        print(f"File size: {os.path.getsize(dest_path) / 1024 / 1024:.2f} MB")
        return True
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

if __name__ == "__main__":
    download_dataset()
