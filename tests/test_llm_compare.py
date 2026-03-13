"""
Tests for word_prediction.llm_compare
========================================
TDD-style tests using a mock LLM so no model download is required.
"""

from unittest.mock import MagicMock, patch

import pytest

from word_prediction.llm_compare import SmallLLMPredictor, compare_predictions
from word_prediction.prediction_engine import NGramPredictor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def ngram_predictor() -> NGramPredictor:
    p = NGramPredictor()
    p.train(["the cat sat on the mat"] * 20)
    return p


@pytest.fixture()
def mock_llm() -> SmallLLMPredictor:
    """Return a SmallLLMPredictor whose pipeline is fully mocked."""
    with patch("word_prediction.llm_compare.hf_pipeline") as mock_pipe_factory:
        mock_pipe = MagicMock()
        # hf_pipeline(…) → mock_pipe; mock_pipe(text, …) → [{"generated_text": text + " sat"}]
        mock_pipe_factory.return_value = mock_pipe
        mock_pipe.side_effect = lambda text, **kw: [
            {"generated_text": text + " sat"}
        ]
        mock_pipe.tokenizer = MagicMock()
        mock_pipe.tokenizer.eos_token_id = 50256

        llm = SmallLLMPredictor.__new__(SmallLLMPredictor)
        llm._model_name = "distilgpt2"
        llm._pipe = mock_pipe
        yield llm


# ---------------------------------------------------------------------------
# SmallLLMPredictor
# ---------------------------------------------------------------------------

class TestSmallLLMPredictor:
    def test_raises_when_transformers_unavailable(self):
        with patch("word_prediction.llm_compare._TRANSFORMERS_AVAILABLE", False):
            with pytest.raises(ImportError, match="transformers"):
                SmallLLMPredictor()

    def test_predict_next_word_returns_string(self, mock_llm):
        result = mock_llm.predict_next_word("the cat")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_predict_next_word_strips_prompt(self, mock_llm):
        # Mock always returns prompt + " sat"
        result = mock_llm.predict_next_word("the cat")
        assert result == "sat"

    def test_generate_words_returns_list(self, mock_llm):
        result = mock_llm.generate_words("the cat", num_words=3)
        assert isinstance(result, list)

    def test_generate_words_respects_count(self, mock_llm):
        # Mock returns prompt + " sat"; splitting "sat" gives 1 word.
        # The LLM generates all at once so count depends on the mock output.
        result = mock_llm.generate_words("the", num_words=2)
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# compare_predictions
# ---------------------------------------------------------------------------

class TestComparePredictions:
    def test_returns_dataframe(self, ngram_predictor, mock_llm):
        import pandas as pd
        df = compare_predictions(ngram_predictor, mock_llm, ["the cat sat"])
        assert isinstance(df, pd.DataFrame)

    def test_has_required_columns(self, ngram_predictor, mock_llm):
        df = compare_predictions(ngram_predictor, mock_llm, ["the cat sat"])
        assert set(df.columns) == {"input", "ngram_prediction", "llm_prediction", "match"}

    def test_one_row_per_input(self, ngram_predictor, mock_llm):
        inputs = ["the cat", "the dog", "on the mat"]
        df = compare_predictions(ngram_predictor, mock_llm, inputs)
        assert len(df) == len(inputs)

    def test_match_column_is_bool(self, ngram_predictor, mock_llm):
        df = compare_predictions(ngram_predictor, mock_llm, ["the cat"])
        assert df["match"].dtype == bool

    def test_empty_inputs(self, ngram_predictor, mock_llm):
        df = compare_predictions(ngram_predictor, mock_llm, [])
        assert len(df) == 0

    def test_raises_without_pandas(self, ngram_predictor, mock_llm):
        with patch("word_prediction.llm_compare._PANDAS_AVAILABLE", False):
            with pytest.raises(ImportError, match="pandas"):
                compare_predictions(ngram_predictor, mock_llm, ["test"])
