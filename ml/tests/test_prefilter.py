import json

from ml.data.prefilter_meta import filter_lines, review_asins


def _line(asin: str, **extra) -> bytes:
    return (json.dumps({"title": "t", **extra, "asin": asin}) + "\n").encode()


def test_keeps_matching_lines_verbatim_in_order_with_duplicates():
    lines = [
        _line("A1", title="first"),
        _line("B2"),
        _line("A1", title="second"),  # duplicate ASIN: kept, after the first
        _line("C3", also_buy=["A1"]),  # mentions A1 only as a value: dropped
        _line("D4"),
    ]
    stats = {}
    out = list(filter_lines(lines, {"A1", "D4"}, stats))
    assert out == [lines[0], lines[2], lines[4]]
    assert stats == {"total": 5, "kept": 3}


def test_review_asins():
    reviews = [json.dumps({"asin": a, "reviewerID": "u"}).encode() for a in ("X", "Y", "X")]
    assert review_asins(reviews) == {"X", "Y"}
