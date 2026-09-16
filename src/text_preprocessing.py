import re
import pandas as pd


def clean_text(text):
    """
    Clean an IT support ticket before feature extraction.
    """

    if pd.isna(text):
        return ""

    text = str(text).lower()

    # Normalize common technical variations
    text = text.replace("wi-fi", "wifi")
    text = text.replace("wi fi", "wifi")
    text = text.replace("e-mail", "email")

    # Expand common IT abbreviations where useful
    text = re.sub(r"\bmfa\b", "multi factor authentication", text)

    # Keep letters, numbers and whitespace
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Normalize multiple spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_dataframe(df, text_column="ticket_text"):
    """
    Apply text preprocessing to a dataframe.
    """

    df = df.copy()

    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in dataset.")

    df["clean_text"] = df[text_column].apply(clean_text)

    return df