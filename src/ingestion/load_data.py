"""
Data ingestion module: Raw JSON → clean tabular data.

This module handles loading, parsing, and cleaning the Amazon Electronics
reviews dataset from gzipped JSON format into a structured dataframe.

Key assumption: A review implies a purchase event.
Rating strength can later be used as signal strength.
"""

import gzip
import json
import pandas as pd
from pathlib import Path
from typing import Optional
from tqdm import tqdm
from datetime import datetime, timezone


def load_reviews(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load reviews from gzipped JSON file and convert to events.
    
    Assumption: A review implies a purchase event.
    Rating strength can later be used as signal strength.
    
    Args:
        path: Path to the .json.gz file
        max_rows: Maximum number of reviews to load (None = load all)
    
    Returns:
        DataFrame with columns: user_id, item_id, event_type, timestamp
    """
    rows = []
    
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f)):
            if max_rows and i >= max_rows:
                break
            
            try:
                data = json.loads(line)
                rows.append({
                    "user_id": data.get("reviewerID"),
                    "item_id": data.get("asin"),
                    "event_type": "purchase",
                    "timestamp": datetime.fromtimestamp(
                        data.get("unixReviewTime"), tz=timezone.utc
                    )
                })
            except json.JSONDecodeError:
                print(f"Warning: Failed to parse line {i}")
                continue
    
    return pd.DataFrame(rows)


def load_items(path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """
    Load product metadata from gzipped JSON file.
    
    Args:
        path: Path to the metadata .json.gz file
        max_rows: Maximum number of items to load (None = load all)
    
    Returns:
        DataFrame with columns: item_id, title, price, category
    """
    rows = []
    skip_count = 0
    
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if max_rows and len(rows) >= max_rows:
                break
            
            # Strip whitespace first
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                data = json.loads(line)
                
                # Extract category safely - it can be a list
                category = data.get("category")
                if isinstance(category, list):
                    category = " > ".join(category) if category else None
                
                rows.append({
                    "item_id": data.get("asin"),
                    "title": data.get("title"),
                    "price": data.get("price"),
                    "category": category,
                })
            except json.JSONDecodeError:
                # Skip malformed lines silently
                skip_count += 1
                continue
            except Exception:
                skip_count += 1
                continue
    
    if skip_count > 0:
        print(f"Note: Skipped {skip_count} malformed lines in metadata file")
    
    return pd.DataFrame(rows)



def save_processed(events_df: pd.DataFrame, items_df: pd.DataFrame):
    """
    Save processed events and items to CSV files.
    
    Args:
        events_df: Events dataframe
        items_df: Items (products) dataframe
    """
    events_df.to_csv("data/processed/events_raw.csv", index=False)
    items_df.to_csv("data/processed/items_raw.csv", index=False)
    print("✓ Saved data/processed/events_raw.csv")
    print("✓ Saved data/processed/items_raw.csv")


def main():
    """Load and save events and items data to processed directory."""
    raw_dir = Path("data/raw")
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Load reviews as events
    print("Loading reviews as events...")
    events_df = load_reviews(str(raw_dir / "reviews_Electronics_5.json.gz"), max_rows=500_000)
    print(f"✓ Loaded {len(events_df):,} purchase events")
    
    print("\nFirst few events:")
    print(events_df.head())
    print("\nEvent data info:")
    print(events_df.info())
    
    # Load items metadata
    print("\n\nLoading product metadata...")
    items_df = load_items(str(raw_dir / "meta_Electronics.json.gz"))
    
    if len(items_df) > 0:
        print(f"✓ Loaded {len(items_df):,} product items")
        print("\nFirst few items:")
        print(items_df.head())
    else:
        print("⚠ No valid items found in metadata file (expected - focusing on events)")
        # Create empty items dataframe with expected schema
        items_df = pd.DataFrame(columns=['item_id', 'title', 'price', 'category'])
    
    # Save processed data
    print("\n\nSaving processed data...")
    save_processed(events_df, items_df)
    print("\n✅ Data processing complete!")


if __name__ == "__main__":
    main()
