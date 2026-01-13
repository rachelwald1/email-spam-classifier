from spam_classifier.model import train_and_evaluate, classify_message


def test_train_and_classify_smoke():
    # Tiny balanced dataset (10 spam, 10 ham) so training + eval works quickly.
    texts = [
        "WIN a free prize now",
        "Claim your voucher today",
        "Limited offer click here to claim",
        "Urgent: you have won a cash prize",
        "Free entry into our prize draw",
        "Congratulations, you have been selected",
        "Act now to claim your reward",
        "Final reminder: claim your prize",
        "Exclusive deal just for you",
        "You have won a free gift card",
        "Are we still on for the meeting tomorrow?",
        "Please review the project update",
        "Can we schedule a call this afternoon?",
        "Here are the notes from yesterday's meeting",
        "Thanks for your help with the report",
        "Let me know when you're available",
        "Looking forward to our discussion tomorrow",
        "Please find the attached agenda",
        "Can you review this draft later today?",
        "I'll send the updated document shortly",
    ]
    labels = [1]*10 + [0]*10  # 1 = spam, 0 = not spam

    result = train_and_evaluate(texts, labels, test_size=0.3, threshold=0.5)

    # Confusion matrix should be 2x2
    assert isinstance(result.confusion, list)
    assert len(result.confusion) == 2
    assert len(result.confusion[0]) == 2
    assert len(result.confusion[1]) == 2

    # Basic inference sanity: returns (label in {0,1}, probability in [0,1])
    label, p = classify_message(result.model, "free prize claim now", threshold=0.5)
    assert label in (0, 1)
    assert 0.0 <= p <= 1.0
