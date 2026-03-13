"""
generate.py
===========
Continuous word-generation loop using a trained NGramPredictor.

The generator appends one predicted word at a time to the growing
context, producing a stream of words until the requested number of words
is reached or the model stops producing predictions.

Typical usage
-------------
>>> from word_prediction.generate import generate_words, word_stream
>>> predictor = NGramPredictor()
>>> predictor.train(["the cat sat on the mat"] * 20)
>>> print(generate_words(predictor, seed="the cat", num_words=5))
['sat', 'on', 'the', 'mat', 'the']
"""

from __future__ import annotations

from typing import Generator, List, Optional

from .data_processing import tokenize
from .prediction_engine import NGramPredictor


def word_stream(
    predictor: NGramPredictor,
    seed: str,
    max_words: int = 50,
) -> Generator[str, None, None]:
    """Yield predicted words one at a time from *seed*.

    The context window grows with every new word, so later predictions
    benefit from the full generated context (up to the model's n-gram
    order).

    Parameters
    ----------
    predictor:
        A trained :class:`~word_prediction.prediction_engine.NGramPredictor`.
    seed:
        Initial text used as the starting context.
    max_words:
        Maximum number of words to generate (default 50).

    Yields
    ------
    str
        One predicted word at a time.

    Raises
    ------
    ValueError
        When *max_words* is less than 1.
    """
    if max_words < 1:
        raise ValueError(f"max_words must be >= 1, got {max_words}")

    tokens: List[str] = tokenize(seed)

    for _ in range(max_words):
        context = " ".join(tokens) if tokens else ""
        next_word: Optional[str] = predictor.predict(context)
        if next_word is None:
            break
        tokens.append(next_word)
        yield next_word


def generate_words(
    predictor: NGramPredictor,
    seed: str,
    num_words: int = 10,
) -> List[str]:
    """Return a list of *num_words* predicted words starting from *seed*.

    This is a convenience wrapper around :func:`word_stream`.

    Parameters
    ----------
    predictor:
        A trained :class:`~word_prediction.prediction_engine.NGramPredictor`.
    seed:
        Initial text used as the starting context.
    num_words:
        Number of words to generate (default 10).

    Returns
    -------
    List[str]
        Ordered list of generated words (may be shorter than *num_words*
        if the model cannot continue).
    """
    return list(word_stream(predictor, seed, max_words=num_words))


def generate_text(
    predictor: NGramPredictor,
    seed: str,
    num_words: int = 10,
) -> str:
    """Return a full generated string: *seed* + *num_words* new words.

    Parameters
    ----------
    predictor:
        A trained :class:`~word_prediction.prediction_engine.NGramPredictor`.
    seed:
        Initial seed text.
    num_words:
        Number of new words to append.

    Returns
    -------
    str
        The seed text followed by the generated words, joined by spaces.
    """
    new_words = generate_words(predictor, seed, num_words)
    parts = [seed.strip()] + new_words
    return " ".join(w for w in parts if w)
