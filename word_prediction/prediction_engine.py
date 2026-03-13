"""
prediction_engine.py
====================
NGramPredictor — Python rewrite of the R ``nextWordPrediction`` function.

The predictor uses a *stupid-backoff* strategy:
  1. Try the 4-gram context  (last 3 words → predict word 4)
  2. Back off to the 3-gram  (last 2 words → predict word 3)
  3. Back off to the 2-gram  (last 1 word  → predict word 2)
  4. Fall back to the most-common unigram
"""

from collections import Counter, defaultdict
from typing import Dict, Optional, Tuple

from .data_processing import build_ngrams, tokenize


class NGramPredictor:
    """N-gram next-word predictor with stupid-backoff.

    Train with :meth:`train` and call :meth:`predict` to get the most likely
    next word for a given input string.

    Examples
    --------
    >>> predictor = NGramPredictor()
    >>> predictor.train(["the cat sat on the mat"] * 10)
    >>> predictor.predict("the cat")  # returns 'sat'
    'sat'
    """

    def __init__(self) -> None:
        # context → {next_word: count}
        self._quad_ctx: Dict[Tuple[str, str, str], Counter] = defaultdict(Counter)
        self._tri_ctx: Dict[Tuple[str, str], Counter] = defaultdict(Counter)
        self._bi_ctx: Dict[str, Counter] = defaultdict(Counter)
        self._unigrams: Counter = Counter()

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, sentences: list) -> None:
        """Train the predictor from a list of sentence strings.

        All previous model data is replaced.

        Parameters
        ----------
        sentences:
            List of raw or pre-cleaned sentence strings.
        """
        self._quad_ctx = defaultdict(Counter)
        self._tri_ctx = defaultdict(Counter)
        self._bi_ctx = defaultdict(Counter)
        self._unigrams = Counter()

        for sentence in sentences:
            tokens = tokenize(sentence)
            if not tokens:
                continue

            # Unigrams
            uni_counts = build_ngrams(tokens, 1)
            for (w,), cnt in uni_counts.items():
                self._unigrams[w] += cnt

            # Bigrams: context = (w1,) → next = w2
            for (w1, w2), cnt in build_ngrams(tokens, 2).items():
                self._bi_ctx[w1][w2] += cnt

            # Trigrams: context = (w1, w2) → next = w3
            for (w1, w2, w3), cnt in build_ngrams(tokens, 3).items():
                self._tri_ctx[(w1, w2)][w3] += cnt

            # Quadgrams: context = (w1, w2, w3) → next = w4
            for (w1, w2, w3, w4), cnt in build_ngrams(tokens, 4).items():
                self._quad_ctx[(w1, w2, w3)][w4] += cnt

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, text: str) -> Optional[str]:
        """Predict the most likely next word for *text*.

        Uses stupid-backoff: tries quadgram context first, then trigram,
        then bigram, then the global most-common unigram.

        Parameters
        ----------
        text:
            Input string (raw or pre-cleaned).

        Returns
        -------
        str or None
            Predicted next word, or ``None`` when the model is untrained.
        """
        tokens = tokenize(text)

        # --- 4-gram backoff (need at least 3 context words) ----------
        if len(tokens) >= 3:
            ctx4 = (tokens[-3], tokens[-2], tokens[-1])
            if ctx4 in self._quad_ctx:
                return self._quad_ctx[ctx4].most_common(1)[0][0]

        # --- 3-gram backoff (need at least 2 context words) ----------
        if len(tokens) >= 2:
            ctx3 = (tokens[-2], tokens[-1])
            if ctx3 in self._tri_ctx:
                return self._tri_ctx[ctx3].most_common(1)[0][0]

        # --- 2-gram backoff (need at least 1 context word) -----------
        if len(tokens) >= 1:
            ctx2 = tokens[-1]
            if ctx2 in self._bi_ctx:
                return self._bi_ctx[ctx2].most_common(1)[0][0]

        # --- unigram fallback ----------------------------------------
        if self._unigrams:
            return self._unigrams.most_common(1)[0][0]

        return None

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def vocabulary_size(self) -> int:
        """Return the number of distinct unigrams in the vocabulary."""
        return len(self._unigrams)

    def is_trained(self) -> bool:
        """Return ``True`` when the predictor has been trained."""
        return bool(self._unigrams)
