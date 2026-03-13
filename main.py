"""
main.py
=======
End-to-end demonstration of the word-prediction pipeline.

Run with:
    python main.py

Or supply a custom corpus file:
    python main.py --corpus path/to/corpus.txt --seed "the quick brown"
"""

from __future__ import annotations

import argparse
import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="N-gram word prediction — end-to-end pipeline demo"
    )
    parser.add_argument(
        "--corpus",
        default=None,
        metavar="PATH",
        help="Path to a plain-text corpus file (one sentence per line). "
             "Defaults to the built-in sample corpus.",
    )
    parser.add_argument(
        "--seed",
        default="the quick brown",
        help="Seed text for continuous word generation (default: 'the quick brown').",
    )
    parser.add_argument(
        "--num-words",
        type=int,
        default=15,
        help="Number of words to generate in the continuous loop (default: 15).",
    )
    parser.add_argument(
        "--compare-llm",
        action="store_true",
        help="Also run the small LLM (DistilGPT-2) comparison. "
             "Requires 'transformers' and an internet connection for the first run.",
    )
    parser.add_argument(
        "--llm-model",
        default="distilgpt2",
        help="Hugging Face model name for the LLM comparison (default: 'distilgpt2').",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ------------------------------------------------------------------
    # 1. Training pipeline
    # ------------------------------------------------------------------
    from word_prediction.train import run_pipeline

    logger.info("=" * 60)
    logger.info("STEP 1 — Training pipeline")
    logger.info("=" * 60)

    model, metrics = run_pipeline(corpus_path=args.corpus)

    print("\n── Training / Validation / Test Metrics ─────────────────────")
    for split, m in metrics.items():
        print(
            f"  {split:5s}  accuracy={m['accuracy']:.3f}  "
            f"correct={m['correct']:4d}  total={m['total']:4d}"
        )
    print()

    # ------------------------------------------------------------------
    # 2. Continuous word generation loop (N-gram)
    # ------------------------------------------------------------------
    from word_prediction.generate import generate_text, word_stream

    logger.info("=" * 60)
    logger.info("STEP 2 — Continuous word generation (N-gram)")
    logger.info("=" * 60)

    print(f"\n── Generating {args.num_words} words from seed: {args.seed!r} ──")
    print(f"  Seed : {args.seed}")
    print(f"  Words:", end=" ", flush=True)
    context_tokens = args.seed.split()
    for word in word_stream(model, args.seed, max_words=args.num_words):
        print(word, end=" ", flush=True)
        context_tokens.append(word)
    print()

    full_text = generate_text(model, args.seed, num_words=args.num_words)
    print(f"\n  Full generated text:\n  {full_text}\n")

    # ------------------------------------------------------------------
    # 3. LLM comparison (optional)
    # ------------------------------------------------------------------
    if args.compare_llm:
        logger.info("=" * 60)
        logger.info("STEP 3 — LLM comparison (model: %s)", args.llm_model)
        logger.info("=" * 60)

        try:
            from word_prediction.llm_compare import (
                SmallLLMPredictor,
                compare_predictions,
            )

            llm = SmallLLMPredictor(model_name=args.llm_model)

            # Generate from the LLM using the same seed
            print(f"\n── LLM generation from seed: {args.seed!r} ──")
            llm_words = llm.generate_words(args.seed, num_words=args.num_words)
            print(f"  LLM words: {' '.join(llm_words)}\n")

            # Side-by-side comparison on a few test prompts
            test_prompts = [
                args.seed,
                "to be or not",
                "the cat sat",
                "natural language processing",
            ]
            print("── Side-by-side next-word comparison ─────────────────────")
            df = compare_predictions(model, llm, test_prompts)
            print(df.to_string(index=False))
            print()
            match_pct = df["match"].mean() * 100
            print(f"  Agreement rate: {match_pct:.1f}%\n")

        except ImportError as exc:
            logger.warning("LLM comparison skipped: %s", exc)
        except Exception as exc:
            logger.error("LLM comparison failed: %s", exc)
    else:
        logger.info("LLM comparison skipped (use --compare-llm to enable).")

    logger.info("Done.")


if __name__ == "__main__":
    main()
