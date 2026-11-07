import pytest

from ml.data.catalog import clean_text, month_start_ts, parse_price_cents
from ml.data.preprocess import final_window


@pytest.mark.parametrize(
    ("raw", "cents"),
    [
        ("$9.99", 999),
        ("$1,299.00", 129900),
        ("$12,345,678.90", 1234567890),
        ("$5", 500),
        ("$5.9", 590),
        (" $3.26 ", 326),
        ("$0.99", 99),
        ("$5.99 - $19.99", None),  # range -> unknown
        ("$4.49 - $172.38", None),
        ("", None),
        (None, None),
        (".a-box-inner{background-color:#fff}#alohaBuyBoxWidget", None),
        ("9.99", None),  # no currency sign
        ("$1,29.00", None),  # malformed grouping
        ("$9.999", None),
    ],
)
def test_parse_price_cents(raw, cents):
    assert parse_price_cents(raw) == cents


def test_clean_text_matches_upstream_rules():
    assert clean_text("<b>Pens &amp; Pencils</b>\n  10 ct ") == "Pens & Pencils 10 ct"
    assert clean_text(None) == ""


def test_month_start_depends_on_time_zone():
    # The reason preprocessing runs with TZ=UTC: review times are UTC midnights.
    utc = month_start_ts("2016-10", "UTC")
    la = month_start_ts("2016-10", "America/Los_Angeles")
    assert utc == 1475280000  # 2016-10-01T00:00:00Z
    assert la - utc == 7 * 3600  # PDT: a review stamped 2016-10-01T00:00Z falls before it


def test_final_window_takes_the_last_range():
    log = (
        "Initial time range: 2017-10 to 2018-11\n"
        "Items count 137 < 3000, expanding time range...\n"
        "New time range: 2016-10 to 2018-11\n"
    )
    assert final_window(log) == ("2016-10", "2018-11")
    assert final_window("Initial time range: 2017-10 to 2018-11\n") == ("2017-10", "2018-11")
