"""
Preprocessing utilities for genre story generation dataset.

This module contains reusable functions for:
- Text cleaning and normalization
- Length filtering and truncation
- Train/validation/test splitting
- Data quality checks
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import re


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    # Optional: Additional cleaning steps
    # text = text.replace('\n', ' ')  # Replace newlines with spaces
    
    return text


def calculate_word_count(text: str) -> int:
    """
    Calculate word count for a text string.
    
    Args:
        text: Text string
        
    Returns:
        Number of words
    """
    return len(str(text).split())


def truncate_text(text: str, max_words: int) -> str:
    """
    Truncate text to maximum number of words.
    
    Args:
        text: Text string to truncate
        max_words: Maximum number of words to keep
        
    Returns:
        Truncated text string
    """
    words = str(text).split()
    return ' '.join(words[:max_words])


def filter_by_length(
    df: pd.DataFrame, 
    text_column: str = 'story',
    min_words: int = 50, 
    max_words: int = None
) -> pd.DataFrame:
    """
    Filter dataframe by text length.
    
    Args:
        df: Input dataframe
        text_column: Name of column containing text
        min_words: Minimum word count (inclusive)
        max_words: Maximum word count (inclusive), None for no upper limit
        
    Returns:
        Filtered dataframe
    """
    df = df.copy()
    
    # Calculate word counts if not already present
    if 'word_count' not in df.columns:
        df['word_count'] = df[text_column].apply(calculate_word_count)
    
    # Apply filters
    mask = df['word_count'] >= min_words
    if max_words is not None:
        mask &= df['word_count'] <= max_words
    
    return df[mask].reset_index(drop=True)


def preprocess_stories(
    df: pd.DataFrame,
    max_words: int = 800,
    min_words: int = 50,
    clean: bool = True
) -> pd.DataFrame:
    """
    Apply full preprocessing pipeline to stories dataset.
    
    Args:
        df: Input dataframe with 'story' and 'genre' columns
        max_words: Maximum words per story (will truncate)
        min_words: Minimum words per story (will filter out)
        clean: Whether to apply text cleaning
        
    Returns:
        Preprocessed dataframe with statistics
    """
    df = df.copy()
    
    print(f"Original dataset size: {len(df)} stories")
    
    # Calculate original word counts
    df['word_count_original'] = df['story'].apply(calculate_word_count)
    
    # Clean text if requested
    if clean:
        print("Cleaning text...")
        df['story'] = df['story'].apply(clean_text)
    
    # Filter by minimum length
    print(f"Filtering stories with < {min_words} words...")
    df = filter_by_length(df, min_words=min_words)
    print(f"After filtering: {len(df)} stories")
    
    # Truncate to max length
    print(f"Truncating stories to max {max_words} words...")
    df['story_truncated'] = df['story'].apply(lambda x: truncate_text(x, max_words))
    df['word_count_processed'] = df['story_truncated'].apply(calculate_word_count)
    
    # Statistics
    stories_truncated = (df['word_count_original'] > max_words).sum()
    print(f"Stories truncated: {stories_truncated} ({stories_truncated/len(df)*100:.1f}%)")
    print(f"Stories kept intact: {len(df) - stories_truncated} ({(len(df)-stories_truncated)/len(df)*100:.1f}%)")
    
    # Rename for final output
    df['story'] = df['story_truncated']
    df['word_count'] = df['word_count_processed']
    
    return df


def stratified_split(
    df: pd.DataFrame,
    stratify_column: str = 'genre',
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataframe into train/val/test with stratification.
    
    Args:
        df: Input dataframe
        stratify_column: Column to stratify by (e.g., 'genre')
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    from sklearn.model_selection import train_test_split
    
    # First split: train vs (val + test)
    train_df, temp_df = train_test_split(
        df,
        test_size=(1 - train_ratio),
        stratify=df[stratify_column],
        random_state=random_state
    )
    
    # Second split: val vs test
    val_size = val_ratio / (val_ratio + test_ratio)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=(1 - val_size),
        stratify=temp_df[stratify_column],
        random_state=random_state
    )
    
    print(f"Split sizes:")
    print(f"  Train: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} ({len(test_df)/len(df)*100:.1f}%)")
    
    return train_df, val_df, test_df


def analyze_genre_distribution(df: pd.DataFrame, genre_column: str = 'genre') -> pd.DataFrame:
    """
    Analyze genre distribution in dataset.
    
    Args:
        df: Input dataframe
        genre_column: Name of genre column
        
    Returns:
        DataFrame with genre statistics
    """
    genre_stats = df[genre_column].value_counts().reset_index()
    genre_stats.columns = ['genre', 'count']
    genre_stats['percentage'] = (genre_stats['count'] / len(df) * 100).round(2)
    
    return genre_stats


def get_preprocessing_stats(df: pd.DataFrame) -> Dict:
    """
    Get comprehensive statistics about preprocessed dataset.
    
    Args:
        df: Preprocessed dataframe
        
    Returns:
        Dictionary of statistics
    """
    stats = {
        'total_stories': len(df),
        'unique_genres': df['genre'].nunique(),
        'word_count_mean': df['word_count'].mean(),
        'word_count_median': df['word_count'].median(),
        'word_count_std': df['word_count'].std(),
        'word_count_min': df['word_count'].min(),
        'word_count_max': df['word_count'].max(),
    }
    
    return stats


def save_splits(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    output_dir: str = "../data"
):
    """
    Save train/val/test splits to CSV files.
    
    Args:
        train_df: Training dataframe
        val_df: Validation dataframe
        test_df: Test dataframe
        output_dir: Directory to save files
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / "train.csv"
    val_path = output_path / "val.csv"
    test_path = output_path / "test.csv"
    
    # Keep only essential columns
    columns_to_keep = ['genre', 'story', 'title', 'word_count']
    
    train_df[columns_to_keep].to_csv(train_path, index=False)
    val_df[columns_to_keep].to_csv(val_path, index=False)
    test_df[columns_to_keep].to_csv(test_path, index=False)
    
    print(f"\nSaved splits to {output_dir}/")
    print(f"  train.csv: {len(train_df)} stories")
    print(f"  val.csv:   {len(val_df)} stories")
    print(f"  test.csv:  {len(test_df)} stories")