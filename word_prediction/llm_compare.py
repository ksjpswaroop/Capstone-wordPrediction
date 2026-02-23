"""
llm_compare.py
==============
Compare N-gram predictions against a small LLM (DistilGPT-2 by default).

The :class:`SmallLLMPredictor` class wraps a Hugging Face ``text-generation``
pipeline. When the ``transformers`` package or a network connection is
unavailable, the class raises ``ImportError`` on construction so callers can
handle it gracefully.

:func:`compare_predictions` runs both models on a list of input prompts and
returns a :class:`pandas.DataFrame` summarising the results.

Typical usage
-------------
>>> from word_prediction.llm_compare import SmallLLMPredictor, compare_predictions
>>> llm = SmallLLMPredictor()            # downloads DistilGPT-2 on first run
>>> ngram_model = ...                    # trained NGramPredictor
>>> df = compare_predictions(ngram_model, llm, ["the cat sat", "to be or"])
>>> print(df)
"""

from __future__ import annotations

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

try:
    import pandas as pd
    _PANDAS_AVAILABLE = True
except ImportError:
    _PANDAS_AVAILABLE = False

try:
    from transformers import pipeline as hf_pipeline
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False


class SmallLLMPredictor:
    """Next-word predictor backed by a small Hugging Face language model.

    Parameters
    ----------
    model_name:
        Hugging Face model identifier (default ``"distilgpt2"``).

    Raises
    ------
    ImportError
        When the ``transformers`` package is not installed.
    """

    def __init__(self, model_name: str = "distilgpt2") -> None:
        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' package is required for SmallLLMPredictor. "
                "Install it with: pip install transformers"
            )
        self._model_name = model_name
        logger.info("Loading LLM pipeline: %s", model_name)
        self._pipe = hf_pipeline(
            "text-generation",
            model=model_name,
            truncation=True,
        )

    def predict_next_word(self, text: str) -> str:
        """Return the single most-likely next word after *text*.

        The model generates one new token; multi-word results are split and
        only the first token is returned.

        Parameters
        ----------
        text:
            Input prompt string.

        Returns
        -------
        str
            Predicted next word (may be empty when the model produces only
            whitespace or punctuation).
        """
        result = self._pipe(
            text,
            max_new_tokens=1,
            num_return_sequences=1,
            pad_token_id=self._pipe.tokenizer.eos_token_id,
        )
        generated: str = result[0]["generated_text"]
        new_part = generated[len(text):]
        tokens = new_part.strip().split()
        return tokens[0] if tokens else ""

    def generate_words(self, seed: str, num_words: int = 10) -> List[str]:
        """Generate *num_words* words starting from *seed*.

        Parameters
        ----------
        seed:
            Seed text.
        num_words:
            Number of new words to produce (default 10).

        Returns
        -------
        List[str]
            Generated word tokens.
        """
        result = self._pipe(
            seed,
            max_new_tokens=num_words,
            num_return_sequences=1,
            pad_token_id=self._pipe.tokenizer.eos_token_id,
        )
        generated: str = result[0]["generated_text"]
        new_part = generated[len(seed):]
        return new_part.strip().split()


def compare_predictions(
    ngram_predictor,
    llm_predictor: SmallLLMPredictor,
    test_inputs: List[str],
) -> "pd.DataFrame":
    """Compare N-gram and LLM next-word predictions side by side.

    Parameters
    ----------
    ngram_predictor:
        A trained :class:`~word_prediction.prediction_engine.NGramPredictor`.
    llm_predictor:
        A :class:`SmallLLMPredictor` instance.
    test_inputs:
        List of input prompt strings.

    Returns
    -------
    pandas.DataFrame
        Columns: ``input``, ``ngram_prediction``, ``llm_prediction``,
        ``match`` (whether both models agree).

    Raises
    ------
    ImportError
        When ``pandas`` is not installed.
    """
    if not _PANDAS_AVAILABLE:
        raise ImportError(
            "The 'pandas' package is required for compare_predictions. "
            "Install it with: pip install pandas"
        )

    rows = []
    for prompt in test_inputs:
        ngram_pred: Optional[str] = ngram_predictor.predict(prompt)
        llm_pred: str = llm_predictor.predict_next_word(prompt)
        rows.append(
            {
                "input": prompt,
                "ngram_prediction": ngram_pred or "",
                "llm_prediction": llm_pred,
                "match": (ngram_pred or "").lower() == llm_pred.lower(),
            }
        )

    return pd.DataFrame(rows, columns=["input", "ngram_prediction", "llm_prediction", "match"])
