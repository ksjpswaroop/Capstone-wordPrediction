"""
data_processing.py
==================
Text-cleaning, tokenisation and n-gram building utilities.
Python rewrite of the text-cleaning logic in predictionEngine.R.
"""

import re
from collections import Counter
from typing import List, Tuple


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Clean *text* for n-gram prediction.

    Steps (mirrors the R ``cleanTextStrInput`` function):
    1. Lowercase
    2. Remove punctuation and digits
    3. Collapse consecutive whitespace to a single space
    4. Strip leading/trailing whitespace

    Parameters
    ----------
    text:
        Raw input string.

    Returns
    -------
    str
        Cleaned string containing only lowercase letters and spaces.
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    """Split *text* into a list of word tokens.

    Parameters
    ----------
    text:
        Pre-cleaned (or raw) input string.

    Returns
    -------
    List[str]
        Ordered list of word tokens; empty list when *text* is blank.
    """
    cleaned = clean_text(text)
    return cleaned.split() if cleaned else []


# ---------------------------------------------------------------------------
# N-gram building
# ---------------------------------------------------------------------------

def build_ngrams(tokens: List[str], n: int) -> Counter:
    """Build a frequency :class:`Counter` of *n*-grams from *tokens*.

    Each n-gram is represented as a :class:`tuple` of *n* strings.

    Parameters
    ----------
    tokens:
        Ordered list of word tokens.
    n:
        N-gram size (≥ 1).

    Returns
    -------
    Counter
        Mapping ``(w1, …, wn) -> frequency``.

    Raises
    ------
    ValueError
        When *n* is less than 1.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return Counter(ngrams)


# ---------------------------------------------------------------------------
# Train / validation / test split
# ---------------------------------------------------------------------------

def split_data(
    sentences: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[List[str], List[str], List[str]]:
    """Split *sentences* into train, validation and test subsets.

    Parameters
    ----------
    sentences:
        List of sentence strings.
    train_ratio:
        Proportion allocated to training (default 0.80).
    val_ratio:
        Proportion allocated to validation (default 0.10).
    test_ratio:
        Proportion allocated to testing (default 0.10).

    Returns
    -------
    Tuple[List[str], List[str], List[str]]
        ``(train, val, test)`` sublists.

    Raises
    ------
    ValueError
        When ratios do not sum to approximately 1.0 or any ratio is negative.
    """
    if any(r < 0 for r in (train_ratio, val_ratio, test_ratio)):
        raise ValueError("All split ratios must be non-negative.")
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-9:
        raise ValueError(
            f"Split ratios must sum to 1.0, got {total:.6f}."
        )

    n = len(sentences)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = sentences[:train_end]
    val = sentences[train_end:val_end]
    test = sentences[val_end:]
    return train, val, test
