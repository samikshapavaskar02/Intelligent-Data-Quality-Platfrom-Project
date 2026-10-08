import pandas as pd


def detect_duplicates(df):

    # Find duplicate rows
    duplicate_rows = df[df.duplicated(keep=False)].copy()

    # Add a status column
    if not duplicate_rows.empty:
        duplicate_rows["Status"] = "Duplicate"

    return duplicate_rows


def get_duplicate_count(df):

    return df.duplicated().sum()