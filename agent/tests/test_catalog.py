from agent.catalog import BM25Index, SearchFilters, brand_allowed, brand_matches, rrf


def test_brand_matching_on_word_prefixes():
    assert brand_matches("Brother Reseller", "Brother")
    assert brand_matches("CANON USA, INC.", "canon")
    assert brand_matches("Zebra", "Zebra Pen")
    assert brand_matches("Hewlett Packard", "HP")
    assert not brand_matches("Pentel", "Pen")
    assert not brand_matches(None, "Pilot")
    assert brand_allowed("Pilot Corporation", ["Pilot"], [])
    assert not brand_allowed("Pilot Corporation", [], ["Pilot"])
    assert brand_allowed(None, [], ["Pilot"])
    assert not brand_allowed(None, ["Pilot"], [])


def test_rrf_sums_reciprocal_ranks_and_breaks_ties_by_id():
    hits = rrf([[3, 1, 2], [1, 4]], limit=10)
    # 1: 1/62 + 1/61; 3: 1/61; 4: 1/62; 2: 1/63
    assert [h.item_id for h in hits] == [1, 3, 4, 2]
    assert hits[0].text_rank == 2 and hits[0].vector_rank == 1
    tie = rrf([[5], [6]], limit=10)
    assert [h.item_id for h in tie] == [5, 6]
    assert rrf([[1, 2, 3]], limit=2)[1].item_id == 2


def test_bm25_weights_title_matches_and_respects_eligibility(items):
    import numpy as np

    index = BM25Index(items)
    everything = np.ones(len(items), dtype=bool)
    top = index.rank("pilot gel pen", everything, 3)
    assert top[0] in (7, 10)
    only_pentel = np.zeros(len(items), dtype=bool)
    only_pentel[index.row_of[9]] = True
    assert index.rank("pilot gel pen", only_pentel, 5) == [9]
    assert index.rank("the and of", everything, 5) == []  # stopwords only


def test_search_filters_price_stock_category_in_the_query(catalog):
    hits = catalog.search(
        "ink cartridge", SearchFilters(category="ink_cartridges", max_price_cents=2500)
    )
    ids = {h.item_id for h in hits}
    assert 3 not in ids  # unknown price never meets a budget
    assert 4 not in ids  # sold out
    assert 5 not in ids  # $27.99
    assert ids == {1, 2, 6}
    assert catalog.last_stats.eligible == 3


def test_search_without_budget_keeps_unknown_prices(catalog):
    ids = {h.item_id for h in catalog.search("ink", SearchFilters(category="ink_cartridges"))}
    assert 3 in ids and 4 not in ids


def test_search_brand_filters_and_modes(catalog):
    f = SearchFilters(category="gel_pens", exclude_brands=["Pilot"])
    assert {h.item_id for h in catalog.search("gel pen", f)} == {8, 9}
    f = SearchFilters(category="gel_pens", brands=["pilot"])
    assert {h.item_id for h in catalog.search("gel pen", f)} == {7, 10}
    text_only = catalog.search("gel pen", SearchFilters(category="gel_pens"), mode="text")
    assert all(h.vector_rank is None for h in text_only)
    vec_only = catalog.search("gel pen", SearchFilters(category="gel_pens"), mode="vector")
    assert all(h.text_rank is None and h.vector_rank for h in vec_only)
    assert len(vec_only) == 4


def test_items_keep_the_requested_order(catalog):
    assert [i.item_id for i in catalog.items([9, 1, 999, 7])] == [9, 1, 7]
    assert [p.item_id for p in catalog.purchases(7, 10)] == [11, 7]  # most recent first
