"""
train.py
========
End-to-end training pipeline: load corpus → split → train → evaluate.

Typical usage
-------------
>>> from word_prediction.train import run_pipeline
>>> model, metrics = run_pipeline("path/to/corpus.txt")
>>> print(metrics)
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

from .data_processing import split_data, tokenize
from .prediction_engine import NGramPredictor

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Sample built-in corpus (used when no external file is supplied)
# ---------------------------------------------------------------------------

SAMPLE_CORPUS: List[str] = [
    "the quick brown fox jumps over the lazy dog",
    "the dog barked at the fox",
    "the fox ran away quickly",
    "a quick brown dog outpaces a lazy fox",
    "the cat sat on the mat",
    "the mat was on the floor",
    "she sells sea shells by the sea shore",
    "how much wood would a woodchuck chuck",
    "if a woodchuck could chuck wood",
    "to be or not to be that is the question",
    "all that glitters is not gold",
    "the quality of mercy is not strained",
    "what light through yonder window breaks",
    "it is the east and juliet is the sun",
    "two roads diverged in a yellow wood",
    "and sorry i could not travel both",
    "i took the one less traveled by",
    "and that has made all the difference",
    "do not go gentle into that good night",
    "rage rage against the dying of the light",
    "in the beginning was the word",
    "the word was with god and the word was god",
    "you shall know the truth and the truth shall set you free",
    "love is patient love is kind",
    "it does not envy it does not boast",
    "when i was a child i spoke as a child",
    "now i know in part but then i shall know fully",
    "the greatest of these is love",
    "ask not what your country can do for you",
    "ask what you can do for your country",
    "we hold these truths to be self evident",
    "that all men are created equal",
    "four score and seven years ago",
    "our fathers brought forth on this continent a new nation",
    "conceived in liberty and dedicated to the proposition",
    "i have a dream that one day this nation will rise up",
    "now is the time to make real the promises of democracy",
    "the only thing we have to fear is fear itself",
    "the buck stops here",
    "speak softly and carry a big stick",
    "machine learning is a subset of artificial intelligence",
    "deep learning uses neural networks with many layers",
    "natural language processing enables computers to understand text",
    "the model was trained on a large corpus of text",
    "word prediction is a fundamental task in natural language processing",
    "n gram models predict the next word based on context",
    "the probability of a word depends on the previous words",
    "language models are used in many applications today",
    "text generation can produce fluent and coherent sentences",
    "the transformer architecture revolutionized natural language processing",
]


# ---------------------------------------------------------------------------
# Corpus loading
# ---------------------------------------------------------------------------

def load_corpus(path: Optional[str] = None) -> List[str]:
    """Load a plain-text corpus from *path* or return the built-in sample.

    Each non-empty line in the file is treated as one sentence.

    Parameters
    ----------
    path:
        Absolute or relative path to a UTF-8 plain-text corpus file.
        Pass ``None`` (default) to use the built-in :data:`SAMPLE_CORPUS`.

    Returns
    -------
    List[str]
        List of sentence strings.
    """
    if path is None:
        logger.info("No corpus path provided — using built-in sample corpus.")
        return list(SAMPLE_CORPUS)

    if not os.path.isfile(path):
        raise FileNotFoundError(f"Corpus file not found: {path!r}")

    with open(path, encoding="utf-8") as fh:
        sentences = [line.strip() for line in fh if line.strip()]

    logger.info("Loaded %d sentences from %r.", len(sentences), path)
    return sentences


# ---------------------------------------------------------------------------
# Model evaluation
# ---------------------------------------------------------------------------

def evaluate(model: NGramPredictor, sentences: List[str]) -> Dict[str, float]:
    """Evaluate *model* on *sentences* and return accuracy metrics.

    For each sentence with at least two tokens, the model predicts the last
    word from all preceding words; a prediction is *correct* when it matches
    the actual last word.

    Parameters
    ----------
    model:
        A trained :class:`~word_prediction.prediction_engine.NGramPredictor`.
    sentences:
        List of sentence strings to evaluate.

    Returns
    -------
    Dict[str, float]
        ``{"accuracy": float, "total": int, "correct": int}``
    """
    total = 0
    correct = 0

    for sentence in sentences:
        tokens = tokenize(sentence)
        if len(tokens) < 2:
            continue

        context = " ".join(tokens[:-1])
        target = tokens[-1]
        prediction = model.predict(context)

        total += 1
        if prediction == target:
            correct += 1

    accuracy = correct / total if total > 0 else 0.0
    return {"accuracy": accuracy, "total": total, "correct": correct}


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_pipeline(
    corpus_path: Optional[str] = None,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
) -> Tuple[NGramPredictor, Dict[str, Dict[str, float]]]:
    """Run the full training pipeline and return the trained model and metrics.

    Steps
    -----
    1. Load corpus (from *corpus_path* or the built-in sample).
    2. Split into train / validation / test sets.
    3. Train :class:`~word_prediction.prediction_engine.NGramPredictor` on
       the training set.
    4. Evaluate on validation and test sets.

    Parameters
    ----------
    corpus_path:
        Path to a plain-text corpus file, or ``None`` for the sample corpus.
    train_ratio:
        Fraction of data used for training (default 0.80).
    val_ratio:
        Fraction of data used for validation (default 0.10).
    test_ratio:
        Fraction of data used for testing (default 0.10).

    Returns
    -------
    Tuple[NGramPredictor, Dict]
        ``(trained_model, {"train": {...}, "val": {...}, "test": {...}})``
    """
    sentences = load_corpus(corpus_path)

    train_sentences, val_sentences, test_sentences = split_data(
        sentences, train_ratio, val_ratio, test_ratio
    )

    logger.info(
        "Split: %d train / %d val / %d test",
        len(train_sentences),
        len(val_sentences),
        len(test_sentences),
    )

    model = NGramPredictor()
    model.train(train_sentences)
    logger.info("Model trained. Vocabulary size: %d", model.vocabulary_size())

    metrics: Dict[str, Dict[str, float]] = {
        "train": evaluate(model, train_sentences),
        "val": evaluate(model, val_sentences),
        "test": evaluate(model, test_sentences),
    }

    logger.info(
        "Metrics — train acc: %.3f | val acc: %.3f | test acc: %.3f",
        metrics["train"]["accuracy"],
        metrics["val"]["accuracy"],
        metrics["test"]["accuracy"],
    )

    return model, metrics
