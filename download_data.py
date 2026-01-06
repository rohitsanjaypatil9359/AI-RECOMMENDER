#!/usr/bin/env python
"""Download Amazon Electronics reviews dataset."""

import urllib.request
import os
from pathlib import Path

# Create data/raw directory
raw_data_dir = Path("data/raw")
raw_data_dir.mkdir(parents=True, exist_ok=True)

# Dataset URLs (from UCSD Amazon Reviews dataset)
urls = {
    "reviews_Electronics_5.json.gz": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz",
    "meta_Electronics.json.gz": "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz",
}

print("Starting download of Amazon Electronics reviews dataset...")
print("This may take a few minutes depending on your connection speed.\n")

for filename, url in urls.items():
    filepath = raw_data_dir / filename
    
    print(f"Downloading {filename}...")
    print(f"URL: {url}")
    
    try:
        urllib.request.urlretrieve(url, filepath)
        file_size_mb = os.path.getsize(filepath) / (1024 * 1024)
        print(f"✓ Downloaded {filename} ({file_size_mb:.2f} MB)\n")
    except Exception as e:
        print(f"✗ Failed to download {filename}: {e}\n")

print("Download complete!")
print(f"Files saved to: {raw_data_dir.absolute()}")
