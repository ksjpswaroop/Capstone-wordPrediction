"""
Tests for word_prediction.generate
=====================================
TDD-style tests for the continuous word-generation loop.
"""

import pytest

from word_prediction.generate import generate_text, generate_words, word_stream
from word_prediction.prediction_engine import NGramPredictor


# ---------------------------------------------------------------------------
# Fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def predictor() -> NGramPredictor:
    p = NGramPredictor()
    p.train(["the cat sat on the mat"] * 30)
    return p


# ---------------------------------------------------------------------------
# word_stream
# ---------------------------------------------------------------------------

class TestWordStream:
    def test_yields_strings(self, predictor):
        words = list(word_stream(predictor, "the cat", max_words=3))
        assert all(isinstance(w, str) for w in words)

    def test_respects_max_words(self, predictor):
        words = list(word_stream(predictor, "the", max_words=5))
        assert len(words) <= 5

    def test_invalid_max_words_raises(self, predictor):
        with pytest.raises(ValueError):
            list(word_stream(predictor, "the", max_words=0))

    def test_empty_seed(self, predictor):
        # Should fall back to unigram and keep producing
        words = list(word_stream(predictor, "", max_words=3))
        assert len(words) <= 3

    def test_is_generator(self, predictor):
        import types
        result = word_stream(predictor, "the", max_words=5)
        assert isinstance(result, types.GeneratorType)

    def test_untrained_predictor_produces_nothing(self):
        p = NGramPredictor()
        words = list(word_stream(p, "hello", max_words=5))
        assert words == []


# ---------------------------------------------------------------------------
# generate_words
# ---------------------------------------------------------------------------

class TestGenerateWords:
    def test_returns_list(self, predictor):
        result = generate_words(predictor, "the cat")
        assert isinstance(result, list)

    def test_length_does_not_exceed_num_words(self, predictor):
        result = generate_words(predictor, "the cat", num_words=4)
        assert len(result) <= 4

    def test_words_are_strings(self, predictor):
        result = generate_words(predictor, "the cat", num_words=3)
        assert all(isinstance(w, str) for w in result)

    def test_continuous_generation(self, predictor):
        # The loop should chain predictions: each word feeds the next
        words = generate_words(predictor, "the cat", num_words=10)
        assert len(words) >= 1


# ---------------------------------------------------------------------------
# generate_text
# ---------------------------------------------------------------------------

class TestGenerateText:
    def test_returns_string(self, predictor):
        result = generate_text(predictor, "the cat", num_words=3)
        assert isinstance(result, str)

    def test_starts_with_seed(self, predictor):
        result = generate_text(predictor, "the cat", num_words=5)
        assert result.startswith("the cat")

    def test_longer_than_seed(self, predictor):
        seed = "the cat"
        result = generate_text(predictor, seed, num_words=3)
        assert len(result.split()) > len(seed.split())

    def test_empty_seed(self, predictor):
        result = generate_text(predictor, "", num_words=3)
        assert isinstance(result, str)
