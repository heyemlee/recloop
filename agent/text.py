"""Text normalisation shared by the catalog search (BM25) and the compatibility matcher.

Device and cartridge models are written many ways in the catalog: `WF-2760`, `WF2760`,
`WF 2760`, `MG3620/MG2220`. PostgreSQL's `english` parser splits these inconsistently
(`HL-2240` becomes 'hl' and '-2240', `MG3620/J4680` one token), so the agent tokenises
itself: every form above yields the joined token ('wf2760') as well as its parts ('wf',
'2760'), on the document side and the query side alike.
"""

from __future__ import annotations

import re

_NON_ALNUM = re.compile(r"[^0-9a-z]+")
# A short letter prefix followed by a number: 'mg3620', 'wf2760', 'j4680', 'm277dw'.
_PREFIX_NUMBER = re.compile(r"^([a-z]{1,4})(\d[0-9a-z]*)$")

# Dropped from queries only (documents keep every token). Shopping filler plus the words that
# express constraints the code applies as filters rather than as search terms.
QUERY_STOPWORDS = frozenset((
    "a", "an", "the", "and", "or", "of", "for", "to", "in", "on", "at", "by", "with", "from",
    "my", "me", "i", "we", "our", "you", "your", "it", "its", "this", "that", "these", "those",
    "is", "are", "be", "was", "were", "am", "do", "does", "need", "needs", "want", "wants",
    "looking", "look", "find", "get", "buy", "bought", "purchase", "purchased", "order",
    "ordered", "some", "any", "please", "can", "could", "would", "like", "show", "give",
    "recommend", "suggest", "something", "thing", "things", "one", "ones", "good", "nice", "best",
    "cheap", "cheaper", "new", "under", "below", "less", "than", "within", "max", "maximum",
    "most", "least", "budget", "around", "about", "up", "no", "more", "over", "dollar", "dollars",
    "buck", "bucks", "usd", "price", "priced", "cost", "costs", "spend", "last", "time",
    "previous", "previously", "before", "again", "same", "different", "another", "other", "else",
    "instead", "work", "works", "working", "compatible", "compatibility", "fits", "fit", "use",
    "uses", "using",
))  # fmt: skip


def words(text: str) -> list[str]:
    """Lower-case alphanumeric runs: 'Epson WF-2760, 4-Pack' -> ['epson', 'wf', '2760', '4', 'pack']."""
    return [w for w in _NON_ALNUM.split(text.lower()) if w]


def stem(word: str) -> str:
    """Strip common English plural endings so 'cartridges' matches 'cartridge'."""
    if not word.isalpha() or len(word) <= 3:
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and not word.endswith(("ss", "us", "is")):
        return word[:-1]
    return word


def tokens(text: str) -> list[str]:
    """Search tokens of `text`: stemmed words plus the joined and split forms of model numbers."""
    ws = words(text)
    out: list[str] = []
    for i, w in enumerate(ws):
        out.append(stem(w))
        m = _PREFIX_NUMBER.match(w)
        if m:
            out.extend(m.groups())  # 'mg3620' -> 'mg', '3620'
        if i + 1 < len(ws) and w.isalpha() and len(w) <= 4 and ws[i + 1][0].isdigit():
            out.append(w + ws[i + 1])  # 'wf', '2760' -> 'wf2760'
    return out


def query_tokens(text: str) -> list[str]:
    """Distinct search tokens of a query, in order, without stopwords."""
    seen: dict[str, None] = {}
    for t in tokens(text):
        if t not in QUERY_STOPWORDS and t not in seen:
            seen[t] = None
    return list(seen)


def norm_phrase(text: str) -> str:
    """'Post-it ' -> 'post it'; used to compare brand names and quoted evidence."""
    return " ".join(words(text))
