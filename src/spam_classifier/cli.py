from __future__ import annotations

import argparse

from spam_classifier.data import load_csv
from spam_classifier.model import (
    train_and_evaluate,
    classify_message,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a spam classifier and classify new messages."
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to labelled CSV dataset (must contain text + label columns).",
    )
    parser.add_argument(
        "--text-col",
        default="text",
        help="Name of the text column in the CSV (default: text).",
    )
    parser.add_argument(
        "--label-col",
        default="label",
        help="Name of the label column in the CSV (default: label).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data held out for testing (default: 0.2).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold for classifying spam (default: 0.5).",
    )
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Train and print evaluation only (skip interactive prompt).",
    )

    args = parser.parse_args()

    # 1) Load dataset
    texts, labels = load_csv(
        args.csv,
        text_column=args.text_col,
        label_column=args.label_col,
    )

    # 2) Train + evaluate
    result = train_and_evaluate(
        texts,
        labels,
        test_size=args.test_size,
        threshold=args.threshold,
    )

    # 3) Print evaluation
    print("\n=== Evaluation on held-out test set ===")
    print("Confusion matrix [[TN, FP],[FN, TP]]:")
    print(result.confusion)
    print("\nClassification report:")
    print(result.report)

    if args.no_interactive:
        return 0

    # 4) Interactive classification loop
    print("\nInteractive mode: paste/type an email. Empty line to quit.\n")
    while True:
        try:
            msg = input("Email text> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not msg:
            print("Bye.")
            return 0

        label, p = classify_message(result.model, msg, threshold=args.threshold)
        verdict = "SPAM" if label == 1 else "NOT SPAM"
        print(f"→ {verdict}  (P(spam)={p:.3f}, threshold={args.threshold})\n")


if __name__ == "__main__":
    raise SystemExit(main())
