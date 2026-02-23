"""
Tests for word_prediction.train
=================================
TDD-style tests for the training pipeline.
"""

import os
import tempfile

import pytest

from word_prediction.train import (
    SAMPLE_CORPUS,
    evaluate,
    load_corpus,
    run_pipeline,
)
from word_prediction.prediction_engine import NGramPredictor


# ---------------------------------------------------------------------------
# load_corpus
# ---------------------------------------------------------------------------

class TestLoadCorpus:
    def test_returns_sample_when_no_path(self):
        corpus = load_corpus()
        assert corpus == SAMPLE_CORPUS

    def test_returns_list_of_strings(self):
        corpus = load_corpus()
        assert isinstance(corpus, list)
        assert all(isinstance(s, str) for s in corpus)

    def test_load_from_file(self, tmp_path):
        p = tmp_path / "corpus.txt"
        p.write_text("hello world\ngoodbye world\n")
        corpus = load_corpus(str(p))
        assert corpus == ["hello world", "goodbye world"]

    def test_skips_empty_lines(self, tmp_path):
        p = tmp_path / "corpus.txt"
        p.write_text("line one\n\n\nline two\n")
        corpus = load_corpus(str(p))
        assert corpus == ["line one", "line two"]

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_corpus("/nonexistent/path/corpus.txt")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    @pytest.fixture()
    def model(self):
        p = NGramPredictor()
        p.train(["the cat sat on the mat"] * 20)
        return p

    def test_returns_dict_with_accuracy_total_correct(self, model):
        metrics = evaluate(model, ["the cat sat on the mat"])
        assert "accuracy" in metrics
        assert "total" in metrics
        assert "correct" in metrics

    def test_accuracy_between_0_and_1(self, model):
        metrics = evaluate(model, ["the cat sat", "on the mat"])
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_empty_sentences_gives_zero_total(self, model):
        metrics = evaluate(model, [])
        assert metrics["total"] == 0
        assert metrics["accuracy"] == 0.0

    def test_short_sentences_skipped(self, model):
        # Single-word sentences have no context → skipped
        metrics = evaluate(model, ["the"])
        assert metrics["total"] == 0

    def test_perfect_prediction(self, model):
        # Model should predict "on" after "the cat sat"
        metrics = evaluate(model, ["the cat sat on"])
        assert metrics["total"] == 1


# ---------------------------------------------------------------------------
# run_pipeline
# ---------------------------------------------------------------------------

class TestRunPipeline:
    def test_returns_model_and_metrics(self):
        model, metrics = run_pipeline()
        assert isinstance(model, NGramPredictor)
        assert isinstance(metrics, dict)

    def test_metrics_keys(self):
        _, metrics = run_pipeline()
        assert set(metrics.keys()) == {"train", "val", "test"}

    def test_each_split_has_accuracy(self):
        _, metrics = run_pipeline()
        for split in ("train", "val", "test"):
            assert "accuracy" in metrics[split]
            assert 0.0 <= metrics[split]["accuracy"] <= 1.0

    def test_model_is_trained(self):
        model, _ = run_pipeline()
        assert model.is_trained()

    def test_custom_corpus_file(self, tmp_path):
        p = tmp_path / "corpus.txt"
        lines = ["the cat sat on the mat\n"] * 50
        p.write_text("".join(lines))
        model, metrics = run_pipeline(str(p))
        assert model.is_trained()
        # Training accuracy should be high for repeated sentences
        assert metrics["train"]["accuracy"] > 0.5

    def test_invalid_ratios_raise(self):
        with pytest.raises(ValueError):
            run_pipeline(train_ratio=0.5, val_ratio=0.5, test_ratio=0.5)
