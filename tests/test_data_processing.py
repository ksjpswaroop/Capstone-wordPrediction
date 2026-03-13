"""
Tests for word_prediction.data_processing
==========================================
TDD-style tests written before (and driving) the implementation.
"""

import pytest

from word_prediction.data_processing import (
    build_ngrams,
    clean_text,
    split_data,
    tokenize,
)


# ---------------------------------------------------------------------------
# clean_text
# ---------------------------------------------------------------------------

class TestCleanText:
    def test_lowercase(self):
        assert clean_text("Hello World") == "hello world"

    def test_removes_punctuation(self):
        assert clean_text("Hello, World!") == "hello world"

    def test_removes_digits(self):
        assert clean_text("test123") == "test"

    def test_collapses_whitespace(self):
        assert clean_text("  too   many   spaces  ") == "too many spaces"

    def test_empty_string(self):
        assert clean_text("") == ""

    def test_only_punctuation(self):
        assert clean_text("!!!???...") == ""

    def test_mixed_case_and_punctuation(self):
        assert clean_text("It's a Beautiful Day!") == "its a beautiful day"


# ---------------------------------------------------------------------------
# tokenize
# ---------------------------------------------------------------------------

class TestTokenize:
    def test_basic_split(self):
        assert tokenize("hello world") == ["hello", "world"]

    def test_cleans_before_split(self):
        assert tokenize("Hello, World!") == ["hello", "world"]

    def test_empty_string(self):
        assert tokenize("") == []

    def test_single_word(self):
        assert tokenize("python") == ["python"]

    def test_whitespace_only(self):
        assert tokenize("   ") == []


# ---------------------------------------------------------------------------
# build_ngrams
# ---------------------------------------------------------------------------

class TestBuildNgrams:
    def test_unigrams(self):
        tokens = ["a", "b", "a"]
        counts = build_ngrams(tokens, 1)
        assert counts[("a",)] == 2
        assert counts[("b",)] == 1

    def test_bigrams(self):
        tokens = ["the", "cat", "sat"]
        counts = build_ngrams(tokens, 2)
        assert counts[("the", "cat")] == 1
        assert counts[("cat", "sat")] == 1

    def test_trigrams(self):
        tokens = ["a", "b", "c", "d"]
        counts = build_ngrams(tokens, 3)
        assert counts[("a", "b", "c")] == 1
        assert counts[("b", "c", "d")] == 1

    def test_n_larger_than_tokens(self):
        tokens = ["a", "b"]
        counts = build_ngrams(tokens, 5)
        assert len(counts) == 0

    def test_n_equals_token_length(self):
        tokens = ["a", "b", "c"]
        counts = build_ngrams(tokens, 3)
        assert counts[("a", "b", "c")] == 1
        assert len(counts) == 1

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            build_ngrams(["a"], 0)

    def test_empty_tokens(self):
        counts = build_ngrams([], 2)
        assert len(counts) == 0


# ---------------------------------------------------------------------------
# split_data
# ---------------------------------------------------------------------------

class TestSplitData:
    def test_default_split_sizes(self):
        sentences = [f"sentence {i}" for i in range(100)]
        train, val, test = split_data(sentences)
        assert len(train) == 80
        assert len(val) == 10
        assert len(test) == 10

    def test_all_in_train(self):
        sentences = list(range(10))
        train, val, test = split_data(sentences, 1.0, 0.0, 0.0)
        assert len(train) == 10
        assert len(val) == 0
        assert len(test) == 0

    def test_no_overlap(self):
        sentences = list(range(20))
        train, val, test = split_data(sentences)
        combined = train + val + test
        assert sorted(combined) == sorted(sentences)

    def test_preserves_order(self):
        sentences = list(range(10))
        train, val, test = split_data(sentences)
        assert train == [0, 1, 2, 3, 4, 5, 6, 7]
        assert val == [8]
        assert test == [9]

    def test_ratios_not_summing_to_one_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_data([], 0.5, 0.5, 0.5)

    def test_negative_ratio_raises(self):
        with pytest.raises(ValueError, match="non-negative"):
            split_data([], 0.8, -0.1, 0.3)

    def test_empty_list(self):
        train, val, test = split_data([])
        assert train == []
        assert val == []
        assert test == []
