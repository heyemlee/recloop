from agent import taxonomy, text
from agent.tests.conftest import INK, NOTEBOOK, STICKY


def test_model_numbers_tokenize_the_same_way_however_written():
    for written in ("WF-2760", "WF2760", "wf 2760", "Epson WF-2760, 4-Pack"):
        assert "wf2760" in text.tokens(written), written
    assert {"mg3620", "mg", "3620", "mg2220"} <= set(text.tokens("MG3620/MG2220"))
    assert "2540" in text.tokens("Deskjet 1000, 1010, 2540")


def test_stemming_is_consistent():
    assert text.stem("cartridges") == text.stem("cartridge") == "cartridge"
    assert text.stem("batteries") == "battery"
    assert text.stem("boxes") == "box"
    assert text.stem("glass") == "glass"
    assert text.stem("2540s") == "2540s"  # not alphabetic: left alone


def test_query_tokens_drop_filler_and_keep_models():
    toks = text.query_tokens("I need ink for my Canon MG3620 under 30 dollars please")
    assert "need" not in toks and "under" not in toks and "dollar" not in toks
    assert {"ink", "canon", "mg3620"} <= set(toks)
    assert len(toks) == len(set(toks))


def test_category_membership_uses_include_and_exclude():
    assert taxonomy.in_category(NOTEBOOK, "notebooks")
    assert not taxonomy.in_category(STICKY, "notebooks")  # sticky notes sit under notebooks
    assert taxonomy.in_category(STICKY, "sticky_notes")
    assert taxonomy.in_category(INK, "ink_cartridges")
    assert not taxonomy.in_category((), "ink_cartridges")
    assert not taxonomy.in_category(None, "pens")


def test_every_category_has_aliases_and_unique_keys():
    assert len(taxonomy.KEYS) == len(set(taxonomy.KEYS))
    for cat in taxonomy.CATEGORIES.values():
        assert cat.include and cat.aliases and cat.label


def test_alias_matching_prefers_the_first_and_longest_phrase():
    assert taxonomy.match_alias("toner for my laser printer")[0] == "toner"
    assert taxonomy.match_alias("printer ink, black")[0] == "ink_cartridges"
    assert taxonomy.match_alias("fountain pen ink in blue")[0] == "bottled_ink"
    assert taxonomy.match_alias("binder clips")[0] == "clips"
    assert taxonomy.match_alias("an ink pad for stamps")[0] == "stamps"
    assert taxonomy.match_alias("gel pens please")[0] == "gel_pens"
    assert taxonomy.match_alias("nothing here") is None
