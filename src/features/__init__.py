"""
Features module: Load and prepare event data for modeling.

This module provides utilities to load cleaned events and prepare them
for feature engineering and model training.
"""

import pandas as pd
from pathlib import Path


# Data paths
RAW_EVENTS_PATH = "data/processed/events_raw.csv"
CLEAN_EVENTS_PATH = "data/processed/events.csv"


def load_events(path: str) -> pd.DataFrame:
    """
    Load events from CSV file.
    
    Args:
        path: Path to events CSV file
    
    Returns:
        DataFrame with columns: user_id, item_id, event_type, timestamp
    """
    df = pd.read_csv(path, parse_dates=["timestamp"])
    return df


def load_clean_events() -> pd.DataFrame:
    """Load cleaned events (single source of truth for modeling)."""
    return load_events(CLEAN_EVENTS_PATH)


def load_raw_events() -> pd.DataFrame:
    """Load raw events (for debugging/comparison)."""
    return load_events(RAW_EVENTS_PATH)


def normalize_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize event_type field: lowercase and strip whitespace.
    
    Real pipelines break because of inconsistent casing/spacing:
    - "Purchase", "purchase", "PURCHASE"
    - "purchase ", " purchase", " purchase "
    
    This future-proofs the pipeline for additional event types (view, cart, etc.)
    
    Args:
        df: Events dataframe
    
    Returns:
        DataFrame with normalized event_type column
    """
    df = df.copy()
    df["event_type"] = df["event_type"].str.lower().str.strip()
    return df


def drop_invalid_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows with missing critical fields.
    
    Silently removes invalid rows without logging or drama.
    Critical fields: user_id, item_id, timestamp
    
    Args:
        df: Events dataframe
    
    Returns:
        DataFrame with invalid rows removed
    """
    df = df.dropna(subset=["user_id", "item_id", "timestamp"])
    return df


if __name__ == "__main__":
    # Quick test
    print("Loading clean events...")
    events = load_clean_events()
    print(f"Loaded {len(events):,} events")
    print(f"Date range: {events['timestamp'].min()} to {events['timestamp'].max()}")
    
    # Test normalization
    print("\n\nTesting event type normalization...")
    # Create test dataframe with inconsistent event types
    test_events = pd.DataFrame({
        "user_id": ["user1", "user2", "user3", "user4"],
        "item_id": ["item1", "item2", "item3", "item4"],
        "event_type": ["Purchase", "purchase ", " PURCHASE", "PURCHASE"],
        "timestamp": pd.date_range("2020-01-01", periods=4)
    })
    
    print("\nBefore normalization:")
    print(test_events["event_type"].values)
    
    normalized = normalize_event_types(test_events)
    print("\nAfter normalization:")
    print(normalized["event_type"].values)
    print("All normalized correctly:", (normalized["event_type"] == "purchase").all())
    
    # Test invalid row dropping
    print("\n\nTesting invalid row dropping...")
    test_events_with_nulls = pd.DataFrame({
        "user_id": ["user1", "user2", None, "user4", "user5"],
        "item_id": ["item1", None, "item3", "item4", "item5"],
        "event_type": ["purchase", "purchase", "purchase", "purchase", "purchase"],
        "timestamp": [pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-02"), 
                      pd.Timestamp("2020-01-03"), None, pd.Timestamp("2020-01-05")]
    })
    
    print(f"\nBefore dropping: {len(test_events_with_nulls)} rows")
    print(test_events_with_nulls)
    
    clean = drop_invalid_rows(test_events_with_nulls)
    print(f"\nAfter dropping invalid rows: {len(clean)} rows")
    print(clean)
    
    print("\n\nFirst few real events:")
    print(events.head())
