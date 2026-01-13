from __future__ import annotations

import pandas as pd


def normalize_labels(labels: pd.Series) -> pd.Series:
    """
    Convert labels into binary integers:
    1 = spam
    0 = not spam
    """
    labels = labels.astype(str).str.strip().str.lower()

    spam_values = {"spam", "1", "true", "yes", "y"}
    ham_values = {"ham", "not spam", "not_spam", "0", "false", "no", "n"}

    def map_label(value: str) -> int:
        if value in spam_values:
            return 1
        if value in ham_values:
            return 0
        raise ValueError(f"Unrecognised label: {value}")

    return labels.map(map_label)


def load_csv(
    path: str,
    *,
    text_column: str = "text",
    label_column: str = "label",
) -> tuple[list[str], list[int]]:
    """
    Load a labelled email dataset from CSV and return texts and labels.
    """
    df = pd.read_csv(path)

    if text_column not in df.columns or label_column not in df.columns:
        raise ValueError(
            f"CSV must contain columns '{text_column}' and '{label_column}'"
        )

    texts = df[text_column].astype(str).fillna("").tolist()
    labels = normalize_labels(df[label_column]).tolist()

    if len(texts) < 20:
        raise ValueError("Dataset too small for training (<20 samples).")

    if len(set(labels)) < 2:
        raise ValueError("Dataset must contain both spam and non-spam examples.")

    return texts, labels
