"""
Tests for word_prediction.prediction_engine
=============================================
TDD-style tests for NGramPredictor.
"""

import pytest

from word_prediction.prediction_engine import NGramPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def trained_predictor() -> NGramPredictor:
    """Return a predictor trained on a small, deterministic corpus."""
    predictor = NGramPredictor()
    corpus = [
        "the cat sat on the mat",
        "the cat ate the rat",
        "the dog sat on the log",
        "the cat sat on the mat",  # repeated to make 'sat' dominate
        "the cat sat on the mat",
    ]
    predictor.train(corpus)
    return predictor


# ---------------------------------------------------------------------------
# is_trained / vocabulary_size
# ---------------------------------------------------------------------------

class TestPredicatorState:
    def test_untrained_is_not_trained(self):
        p = NGramPredictor()
        assert not p.is_trained()

    def test_trained_is_trained(self, trained_predictor):
        assert trained_predictor.is_trained()

    def test_vocabulary_size_positive(self, trained_predictor):
        assert trained_predictor.vocabulary_size() > 0

    def test_retrain_resets_state(self, trained_predictor):
        vocab_before = trained_predictor.vocabulary_size()
        trained_predictor.train(["one two three"])
        assert trained_predictor.vocabulary_size() < vocab_before


# ---------------------------------------------------------------------------
# predict – bigram backoff
# ---------------------------------------------------------------------------

class TestBigramPrediction:
    def test_predicts_from_single_word(self, trained_predictor):
        # "the" is most often followed by "cat" or "mat" – just check non-None
        result = trained_predictor.predict("the")
        assert result is not None
        assert isinstance(result, str)

    def test_known_bigram_context(self, trained_predictor):
        # after "sat" the word "on" appears consistently
        result = trained_predictor.predict("sat")
        assert result == "on"


# ---------------------------------------------------------------------------
# predict – trigram backoff
# ---------------------------------------------------------------------------

class TestTrigramPrediction:
    def test_trigram_context(self, trained_predictor):
        # "cat sat" → "on" (dominant)
        result = trained_predictor.predict("cat sat")
        assert result == "on"


# ---------------------------------------------------------------------------
# predict – quadgram backoff
# ---------------------------------------------------------------------------

class TestQuadgramPrediction:
    def test_quadgram_context(self, trained_predictor):
        # "cat sat on" → "the"
        result = trained_predictor.predict("cat sat on")
        assert result == "the"

    def test_long_context_uses_last_three_words(self, trained_predictor):
        # extra leading words should not matter
        result_short = trained_predictor.predict("cat sat on")
        result_long = trained_predictor.predict("blah blah blah cat sat on")
        assert result_short == result_long


# ---------------------------------------------------------------------------
# predict – edge cases
# ---------------------------------------------------------------------------

class TestPredictEdgeCases:
    def test_empty_input_returns_most_common_word(self, trained_predictor):
        result = trained_predictor.predict("")
        assert result is not None

    def test_unknown_word_falls_back_to_unigram(self, trained_predictor):
        result = trained_predictor.predict("zxqvbnm")
        assert result is not None  # falls back to most common unigram

    def test_untrained_predictor_returns_none(self):
        p = NGramPredictor()
        assert p.predict("hello world") is None

    def test_punctuation_in_input_is_ignored(self, trained_predictor):
        result_clean = trained_predictor.predict("cat sat")
        result_dirty = trained_predictor.predict("cat, sat!")
        assert result_clean == result_dirty
