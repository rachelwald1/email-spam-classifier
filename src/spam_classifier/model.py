"""
End-to-end text classification model:
raw text → TF-IDF features → logistic regression → probabilistic output.

Designed to emphasise interpretability, validation on unseen data,
and explicit handling of uncertainty via a configurable decision threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix


@dataclass
class TrainResult:
    """
    Returned after training so you can inspect performance and reuse the trained model.
    """
    model: Pipeline
    threshold: float
    confusion: list[list[int]]
    report: str


def build_model() -> Pipeline:
    """
    Build an end-to-end pipeline:
      raw text -> TF-IDF features -> logistic regression classifier
    """
    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    stop_words="english",
                    ngram_range=(1, 2),  # unigrams + bigrams
                    min_df=2,            # ignore extremely rare tokens
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_and_evaluate(
    texts: list[str],
    labels: list[int],
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.5,
) -> TrainResult:
    """
    Train on a dataset and evaluate on held-out data.

    - Splits the data into train/test.
    - Fits the pipeline on the train set.
    - Predicts probabilities on the test set.
    - Applies a decision threshold to convert probabilities -> class labels.
    - Produces a confusion matrix + precision/recall report.
    """
    if len(texts) != len(labels):
        raise ValueError("texts and labels must have the same length.")
    if len(set(labels)) < 2:
        raise ValueError("Need both classes (spam and not spam) to train.")

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = build_model()
    model.fit(X_train, y_train)

    # Predict spam probabilities for the held-out test set
    p_spam = model.predict_proba(X_test)[:, 1]

    # Convert probabilities to class predictions using your chosen threshold
    y_pred = (p_spam >= threshold).astype(int)

    conf = confusion_matrix(y_test, y_pred).tolist()
    rep = classification_report(y_test, y_pred, target_names=["not_spam", "spam"])

    return TrainResult(model=model, threshold=threshold, confusion=conf, report=rep)


def predict_spam_probability(model: Pipeline, message: str) -> float:
    """
    Compute P(spam) for a single new message.
    """
    return float(model.predict_proba([message])[0][1])


def classify_message(model: Pipeline, message: str, *, threshold: float = 0.5) -> tuple[int, float]:
    """
    Return (label, probability) where:
      label: 1 = spam, 0 = not spam
      probability: P(spam)
    """
    p = predict_spam_probability(model, message)
    label = 1 if p >= threshold else 0
    return label, p
