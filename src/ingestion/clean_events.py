"""
Data cleaning pipeline: events_raw.csv → events.csv

Single responsibility: Transform raw events into clean, consistent, reliable data
that becomes the single source of truth for all downstream models.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from features import load_events, clean_events as clean_events_util


def load_raw_events(path: str) -> pd.DataFrame:
    """Load raw events from CSV."""
    df = pd.read_csv(path, parse_dates=["timestamp"])
    print(f"Loaded {len(df):,} raw events")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate events (same user, item, timestamp).
    
    Duplicates can arise from data collection errors or user behavior
    (e.g., multiple review submissions).
    """
    initial_count = len(df)
    
    # Deduplicate on (user_id, item_id, timestamp)
    # Keep first occurrence to preserve temporal order
    df = df.drop_duplicates(subset=["user_id", "item_id", "timestamp"], keep="first")
    
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"Removed {dropped:,} duplicate events")
    
    return df


def remove_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove events with missing critical fields.
    
    All four fields are required: user_id, item_id, event_type, timestamp.
    """
    initial_count = len(df)
    
    df = df.dropna(subset=["user_id", "item_id", "event_type", "timestamp"])
    
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"Removed {dropped:,} events with null values")
    
    return df


def validate_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure correct data types and convert where needed.
    """
    # user_id and item_id should be strings
    df["user_id"] = df["user_id"].astype(str)
    df["item_id"] = df["item_id"].astype(str)
    
    # event_type should be string
    df["event_type"] = df["event_type"].astype(str)
    
    # timestamp should be datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    return df


def validate_event_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure event_type has only valid values.
    
    For now, only "purchase" events are supported.
    """
    valid_types = ["purchase"]
    
    initial_count = len(df)
    df = df[df["event_type"].isin(valid_types)]
    
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"Removed {dropped:,} events with invalid event_type")
    
    return df


def remove_outliers(df: pd.DataFrame, user_min_interactions: int = 1) -> pd.DataFrame:
    """
    Remove users with suspiciously low activity.
    
    This filters out single-interaction users who may be bot activity or
    one-time random clicks. Keeps only users with meaningful engagement.
    
    Args:
        df: Events dataframe
        user_min_interactions: Minimum interactions per user
    """
    initial_count = len(df)
    
    # Count interactions per user
    user_counts = df.groupby("user_id").size()
    
    # Filter to users with >= min interactions
    valid_users = user_counts[user_counts >= user_min_interactions].index
    df = df[df["user_id"].isin(valid_users)]
    
    dropped = initial_count - len(df)
    if dropped > 0:
        print(f"Removed {dropped:,} events from users with <{user_min_interactions} interactions")
    
    return df


def sort_and_reset_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sort by timestamp and reset index for consistent output.
    
    This ensures temporal consistency: events are in chronological order.
    """
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def generate_stats(df: pd.DataFrame) -> dict:
    """Generate summary statistics about cleaned events."""
    return {
        "total_events": len(df),
        "unique_users": df["user_id"].nunique(),
        "unique_items": df["item_id"].nunique(),
        "date_range": (df["timestamp"].min(), df["timestamp"].max()),
        "events_per_user_mean": df.groupby("user_id").size().mean(),
        "events_per_user_median": df.groupby("user_id").size().median(),
    }


def clean_events(input_path: str, output_path: str) -> dict:
    """
    Main cleaning pipeline.
    
    Args:
        input_path: Path to raw events CSV
        output_path: Path to save cleaned events CSV
    
    Returns:
        Dictionary with cleaning statistics
    """
    print("="*60)
    print("DATA CLEANING PIPELINE")
    print("="*60)
    
    # Load raw data
    print("\n[1/7] Loading raw events...")
    df = load_raw_events(input_path)
    
    # Remove nulls
    print("[2/7] Removing null values...")
    df = remove_nulls(df)
    
    # Validate and convert types
    print("[3/7] Validating data types...")
    df = validate_types(df)
    
    # Validate event types
    print("[4/7] Validating event types...")
    df = validate_event_types(df)
    
    # Remove duplicates
    print("[5/7] Removing duplicates...")
    df = remove_duplicates(df)
    
    # Remove low-activity users
    print("[6/7] Filtering low-activity users...")
    df = remove_outliers(df, user_min_interactions=1)
    
    # Sort by timestamp
    print("[7/7] Sorting by timestamp...")
    df = sort_and_reset_index(df)
    
    # Generate statistics
    stats = generate_stats(df)
    
    # Save cleaned data
    print("\n" + "="*60)
    print("SAVING CLEANED DATA")
    print("="*60)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Saved {len(df):,} clean events to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CLEANING SUMMARY")
    print("="*60)
    print(f"Total events: {stats['total_events']:,}")
    print(f"Unique users: {stats['unique_users']:,}")
    print(f"Unique items: {stats['unique_items']:,}")
    print(f"Date range: {stats['date_range'][0]} to {stats['date_range'][1]}")
    print(f"Mean events/user: {stats['events_per_user_mean']:.2f}")
    print(f"Median events/user: {stats['events_per_user_median']:.0f}")
    print("="*60)
    
    return stats


def main():
    """Run the cleaning pipeline using utilities from src/features."""
    raw_path = "data/processed/events_raw.csv"
    clean_path = "data/processed/events.csv"
    
    # Load raw events
    df = load_events(raw_path)
    
    # Apply complete cleaning pipeline
    cleaned_df = clean_events_util(df)
    
    # Save cleaned events
    cleaned_df.to_csv(clean_path, index=False)
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE")
    print("="*60)
    print(f"Raw events:     {len(df):,}")
    print(f"Clean events:   {len(cleaned_df):,}")
    print(f"Reduction:      {100 * (1 - len(cleaned_df)/len(df)):.1f}%")
    print(f"\nSaved to: {clean_path}")
    print("\nFirst 5 clean events:")
    print(cleaned_df.head())
    print("\nEvent type distribution:")
    print(cleaned_df["event_type"].value_counts())


if __name__ == "__main__":
    main()
