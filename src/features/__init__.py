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


def filter_low_activity_users(df: pd.DataFrame, min_events: int = 3) -> pd.DataFrame:
    """
    Filter out users with fewer than min_events interactions.
    
    Why this matters:
    - Cold-start users (< 3 events) will be handled differently in production
    - Collaborative filtering needs minimum signal to work
    - Low-activity users contribute noise, not signal
    
    Args:
        df: Events dataframe
        min_events: Minimum number of events per user (default: 3)
    
    Returns:
        DataFrame with only active users
    """
    df = df.copy()
    user_counts = df["user_id"].value_counts()
    valid_users = user_counts[user_counts >= min_events].index
    return df[df["user_id"].isin(valid_users)]


def filter_rare_items(df: pd.DataFrame, min_events: int = 5) -> pd.DataFrame:
    """
    Filter out items with fewer than min_events interactions.
    
    Why this rule exists:
    - Reduces noise: Items with 1-2 interactions are data noise
    - Improves model stability: Models work better with more signal
    - Mirrors real-world production: E-commerce platforms filter rare SKUs
    - Cold-start items: New items handled separately (content-based fallback)
    
    Args:
        df: Events dataframe
        min_events: Minimum number of events per item (default: 5)
    
    Returns:
        DataFrame with only sufficiently popular items
    """
    df = df.copy()
    item_counts = df["item_id"].value_counts()
    valid_items = item_counts[item_counts >= min_events].index
    return df[df["item_id"].isin(valid_items)]


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort events chronologically by timestamp.
    
    Non-negotiable for recommender systems:
    - Time-awareness enables temporal evaluation splits (train/test by date)
    - Prevents data leakage (future events bleeding into past)
    - Supports session-based recommendations
    - Enables next-item prediction (temporal dynamics)
    - Required for A/B testing validation
    
    Args:
        df: Events dataframe
    
    Returns:
        DataFrame sorted by timestamp with reset index
    """
    return df.sort_values("timestamp").reset_index(drop=True)


def clean_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Complete event cleaning pipeline.
    
    Composes all cleaning utilities into a single, reproducible process:
    1. Normalize event types (case/whitespace)
    2. Drop invalid rows (missing critical fields)
    3. Filter low-activity users (< 3 events → cold-start)
    4. Filter rare items (< 5 events → noisy)
    5. Sort by time (non-negotiable for temporal splits)
    
    Args:
        df: Raw events dataframe
    
    Returns:
        Clean, production-ready events dataframe
    """
    df = normalize_event_types(df)
    df = drop_invalid_rows(df)
    df = filter_low_activity_users(df, min_events=3)
    df = filter_rare_items(df, min_events=5)
    df = sort_by_time(df)
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
    
    # Test low-activity user filtering
    print("\n\nTesting low-activity user filtering...")
    test_events_users = pd.DataFrame({
        "user_id": ["user1", "user1", "user1", "user2", "user2", "user3"],
        "item_id": ["item1", "item2", "item3", "item4", "item5", "item6"],
        "event_type": ["purchase"] * 6,
        "timestamp": pd.date_range("2020-01-01", periods=6)
    })
    
    print(f"\nBefore filtering: {len(test_events_users)} rows")
    print("User activity:")
    print(test_events_users["user_id"].value_counts().sort_index())
    print("  - user1: 3 events")
    print("  - user2: 2 events (< 3, will be filtered)")
    print("  - user3: 1 event (< 3, will be filtered)")
    
    filtered = filter_low_activity_users(test_events_users, min_events=3)
    print(f"\nAfter filtering (min_events=3): {len(filtered)} rows")
    print("Remaining users:")
    print(filtered["user_id"].value_counts().sort_index())
    
    # Test rare item filtering
    print("\n\nTesting rare item filtering...")
    test_events_items = pd.DataFrame({
        "user_id": ["user1", "user2", "user3", "user4", "user5", "user6"],
        "item_id": ["item1", "item1", "item1", "item1", "item1", "item2"],
        "event_type": ["purchase"] * 6,
        "timestamp": pd.date_range("2020-01-01", periods=6)
    })
    
    print(f"\nBefore filtering: {len(test_events_items)} rows")
    print("Item activity:")
    print(test_events_items["item_id"].value_counts().sort_index())
    print("  - item1: 5 events")
    print("  - item2: 1 event (< 5, will be filtered)")
    
    filtered_items = filter_rare_items(test_events_items, min_events=5)
    print(f"\nAfter filtering (min_events=5): {len(filtered_items)} rows")
    print("Remaining items:")
    print(filtered_items["item_id"].value_counts().sort_index())
    
    # Test temporal sorting
    print("\n\nTesting temporal sorting...")
    test_events_time = pd.DataFrame({
        "user_id": ["user1", "user2", "user3", "user4"],
        "item_id": ["item1", "item2", "item3", "item4"],
        "event_type": ["purchase"] * 4,
        "timestamp": [
            pd.Timestamp("2020-01-05"),  # Out of order
            pd.Timestamp("2020-01-02"),  # Out of order
            pd.Timestamp("2020-01-01"),  # Out of order
            pd.Timestamp("2020-01-03"),  # Out of order
        ]
    })
    
    print(f"\nBefore sorting:")
    print(test_events_time[["timestamp", "user_id"]])
    
    sorted_time = sort_by_time(test_events_time)
    print(f"\nAfter sorting by timestamp:")
    print(sorted_time[["timestamp", "user_id"]])
    is_sorted = (sorted_time["timestamp"].diff().dropna() >= pd.Timedelta(0)).all()
    print("\nTimestamps in chronological order:", is_sorted)
    
    # Test complete pipeline
    print("\n\n" + "="*60)
    print("Testing complete clean_events() pipeline")
    print("="*60)
    
    test_full_pipeline = pd.DataFrame({
        "user_id": ["user1", "user1", "user1", "user2", "user2", "user3", "user4", "user4"],
        "item_id": ["item1", "item1", "item2", "item3", "item3", "item4", "item5", "item6"],
        "event_type": ["Purchase", "purchase ", " PURCHASE", "purchase", "PURCHASE", "purchase", "purchase", "purchase"],
        "timestamp": [
            pd.Timestamp("2020-01-05"),
            pd.Timestamp("2020-01-02"),
            pd.Timestamp("2020-01-01"),
            pd.Timestamp("2020-01-03"),
            pd.Timestamp("2020-01-04"),
            pd.Timestamp("2020-01-06"),
            pd.Timestamp("2020-01-07"),
            pd.Timestamp("2020-01-08"),
        ]
    })
    
    print(f"\nBefore pipeline: {len(test_full_pipeline)} rows")
    print("Users:", test_full_pipeline["user_id"].unique())
    print("Items:", test_full_pipeline["item_id"].unique())
    
    cleaned = clean_events(test_full_pipeline)
    
    print(f"\nAfter pipeline: {len(cleaned)} rows")
    print("Users:", cleaned["user_id"].unique())
    print("Items:", cleaned["item_id"].unique())
    print("\nWhy rows were removed:")
    print("  - user2: only 2 events (< 3 min_events)")
    print("  - user3: only 1 event (< 3 min_events)")
    print("  - item4: only 1 event (< 5 min_events)")
    print("  - item5: only 1 event (< 5 min_events)")
    print("  - item6: only 1 event (< 5 min_events)")
    
    print(f"\nFinal dataset:")
    print(cleaned)
    
    print("\n\nFirst few real events (already sorted):")
    print(events.head())
