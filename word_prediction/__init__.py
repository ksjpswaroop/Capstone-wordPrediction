"""
Word Prediction package — Python rewrite of the R N-gram prediction engine.

Modules:
    data_processing   – text cleaning and n-gram building
    prediction_engine – NGramPredictor (stupid-backoff n-gram model)
    train             – train / validation / test pipeline
    generate          – continuous word-generation loop
    llm_compare       – compare n-gram output against a small LLM
"""

from .data_processing import clean_text, tokenize, build_ngrams, split_data
from .prediction_engine import NGramPredictor

__all__ = [
    "clean_text",
    "tokenize",
    "build_ngrams",
    "split_data",
    "NGramPredictor",
]
