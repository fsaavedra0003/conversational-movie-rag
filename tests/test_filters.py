from app.utils.filters import remove_seen_or_disliked


def test_remove_seen_or_disliked():
    candidates = ["Collateral", "Driven", "The Bourne Identity"]
    seen = {"The Bourne Identity"}
    disliked = {"Driven"}

    result = remove_seen_or_disliked(candidates, seen, disliked)

    assert result == ["Collateral"]