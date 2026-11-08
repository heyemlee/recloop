"""RecLoop shopping assistant (design §10).

The model understands and explains; the code decides. A language model turns the shopper's
words into a `ShoppingRequest` (validated, one retry), may judge device compatibility (with
quoted evidence that the code verifies) and writes one-line reasons. Everything a shopper can
hold the store to (price, stock, category, brand) is checked by code on database rows, and the
reply is rendered from those rows.
"""
