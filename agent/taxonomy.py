"""The categories the assistant can filter on, mapped onto the catalog's category paths.

`items.categories` holds the full Amazon path (`Office Products > Office & School Supplies >
Writing & Correction Supplies > Pens & Refills > Gel Ink Rollerball Pens`), so a category is a
set of path names: an item belongs to it when its path contains one of `include` and none of
`exclude`. Items without a path (65 of 3,459) belong to no category: an unknown category does
not satisfy a category constraint, like an unknown price does not satisfy a budget.

`aliases` are the phrases the rule-based parser of the no-LLM baseline looks for (stemmed,
whole words); the language model gets the keys and labels instead.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from agent import text


@dataclass(frozen=True)
class Category:
    key: str
    label: str
    include: tuple[str, ...]
    exclude: tuple[str, ...] = ()
    aliases: tuple[str, ...] = ()


def _c(key: str, label: str, include, exclude=(), aliases=()) -> Category:
    return Category(key, label, tuple(include), tuple(exclude), tuple(aliases))


# Order matters only for the baseline's alias matching: among aliases of the same length, the
# first category wins (so 'ink' alone means printer ink, the larger category).
_ALL = [
    _c(
        "ink_cartridges",
        "printer ink cartridges",
        ["Inkjet Printer Ink"],
        aliases=[
            "ink",
            "ink cartridge",
            "printer ink",
            "inkjet cartridge",
            "ink tank",
            "cartridge",
        ],
    ),
    _c(
        "toner",
        "laser printer toner and drums",
        ["Laser Printer Drums & Toner"],
        aliases=["toner", "toner cartridge", "laser toner", "drum unit"],
    ),
    _c(
        "printers",
        "printers",
        ["Printers"],
        aliases=["printer", "inkjet printer", "laser printer", "all in one printer"],
    ),
    _c("photo_paper", "photo paper", ["Photo Paper"], aliases=["photo paper"]),
    _c(
        "copy_paper",
        "copy and printer paper",
        ["Copy & Multipurpose Paper"],
        aliases=["copy paper", "printer paper", "printing paper", "multipurpose paper"],
    ),
    _c("card_stock", "card stock", ["Card Stock"], aliases=["card stock", "cardstock"]),
    _c("index_cards", "index cards", ["Index Cards"], aliases=["index card"]),
    _c(
        "notebooks",
        "notebooks and writing pads",
        ["Notebooks & Writing Pads"],
        exclude=["Self-Stick Notes"],
        aliases=["notebook", "notepad", "writing pad", "legal pad", "composition book", "journal"],
    ),
    _c(
        "sticky_notes",
        "sticky notes",
        ["Self-Stick Notes"],
        aliases=["sticky note", "post it note", "self stick note"],
    ),
    _c(
        "ballpoint_pens",
        "ballpoint pens",
        ["Ballpoint Pens"],
        aliases=["ballpoint pen", "ballpoint", "ball point pen"],
    ),
    _c("gel_pens", "gel pens", ["Gel Ink Rollerball Pens"], aliases=["gel pen", "gel ink pen"]),
    _c(
        "rollerball_pens",
        "rollerball pens (gel pens included)",
        ["Rollerball Pens"],
        aliases=["rollerball pen", "rollerball", "roller ball pen"],
    ),
    _c("fountain_pens", "fountain pens", ["Fountain Pens"], aliases=["fountain pen"]),
    _c(
        "pens",
        "pens of any kind (not refills or bottled ink)",
        ["Pens & Refills"],
        exclude=["Bottled Ink", "Pen Refills"],
        aliases=["pen"],
    ),
    _c("pen_refills", "pen refills", ["Pen Refills"], aliases=["pen refill", "refill"]),
    _c(
        "bottled_ink",
        "bottled ink for fountain pens",
        ["Bottled Ink"],
        aliases=["bottled ink", "fountain pen ink", "ink bottle", "bottle of ink"],
    ),
    _c(
        "pencils",
        "wood pencils",
        ["Wooden Lead Pencils"],
        aliases=["pencil", "wood pencil", "wooden pencil", "graphite pencil"],
    ),
    _c(
        "mechanical_pencils",
        "mechanical pencils",
        ["Mechanical Pencils"],
        aliases=["mechanical pencil"],
    ),
    _c(
        "pencil_lead",
        "mechanical pencil lead refills",
        ["Mechanical Pencil Leads"],
        aliases=["pencil lead", "lead refill", "mechanical pencil lead"],
    ),
    _c(
        "colored_pencils",
        "colored pencils",
        ["Wooden Colored Pencils"],
        aliases=["colored pencil", "coloured pencil", "color pencil"],
    ),
    _c("highlighters", "highlighters", ["Highlighters"], aliases=["highlighter"]),
    _c(
        "permanent_markers",
        "permanent markers",
        ["Permanent Markers"],
        aliases=["permanent marker", "marker"],
    ),
    _c(
        "dry_erase_markers",
        "dry erase markers",
        ["Erasable Markers"],
        aliases=["dry erase marker", "whiteboard marker", "erasable marker"],
    ),
    _c(
        "erasers",
        "erasers and correction tape",
        ["Erasers & Correction Products"],
        aliases=["eraser", "correction tape", "white out", "correction fluid"],
    ),
    _c(
        "pencil_sharpeners",
        "pencil sharpeners",
        ["Pencil Sharpeners"],
        aliases=["pencil sharpener", "sharpener"],
    ),
    _c(
        "staplers",
        "staplers",
        ["Staplers & Punches"],
        exclude=["Staples", "Staple Removers"],
        aliases=["stapler"],
    ),
    _c("staples", "staples", ["Staples"], aliases=["staple"]),
    _c("binders", "binders", ["Binders"], aliases=["binder", "ring binder"]),
    _c(
        "sheet_protectors",
        "sheet protectors and sleeves",
        ["Sheet Protectors, Card & Photo Sleeves"],
        aliases=["sheet protector", "page protector"],
    ),
    _c(
        "folders",
        "file folders",
        ["Folders"],
        aliases=["folder", "file folder", "hanging folder", "manila folder"],
    ),
    _c("envelopes", "envelopes", ["Envelopes"], aliases=["envelope"]),
    _c(
        "mailers",
        "mailers",
        ["Mailers"],
        aliases=["mailer", "bubble mailer", "padded envelope", "poly mailer"],
    ),
    _c(
        "packing_tape",
        "packing tape",
        ["Packing Tape"],
        aliases=["packing tape", "shipping tape", "packaging tape"],
    ),
    _c(
        "tape",
        "office tape",
        ["Tape"],
        aliases=["tape", "transparent tape", "clear tape", "scotch tape"],
    ),
    _c("tape_dispensers", "tape dispensers", ["Tape Dispensers"], aliases=["tape dispenser"]),
    _c(
        "glue",
        "glue and adhesives",
        ["Adhesives & Glue", "Adhesives"],
        aliases=["glue", "glue stick", "adhesive"],
    ),
    _c(
        "clips",
        "binder clips and paper clips",
        ["Clips"],
        aliases=["clip", "binder clip", "paper clip"],
    ),
    _c("rubber_bands", "rubber bands", ["Rubber Bands"], aliases=["rubber band"]),
    _c(
        "labels",
        "labels and stickers",
        ["Labels & Stickers"],
        aliases=["label", "sticker", "address label", "shipping label"],
    ),
    _c(
        "label_makers",
        "label makers",
        ["Label Makers"],
        aliases=["label maker", "label printer", "labeler"],
    ),
    _c("shredders", "paper shredders", ["Shredders"], aliases=["shredder", "paper shredder"]),
    _c("calculators", "calculators", ["Calculators"], aliases=["calculator"]),
    _c(
        "planners",
        "planners",
        ["Planners"],
        aliases=["planner", "day planner", "weekly planner", "agenda"],
    ),
    _c(
        "wall_calendars",
        "wall calendars",
        ["Wall Calendars"],
        aliases=["calendar", "wall calendar"],
    ),
    _c(
        "dry_erase_boards",
        "whiteboards",
        ["Dry Erase Boards"],
        aliases=["whiteboard", "white board", "dry erase board"],
    ),
    _c(
        "bulletin_boards",
        "bulletin boards",
        ["Bulletin Boards"],
        aliases=["bulletin board", "cork board", "corkboard"],
    ),
    _c("laminators", "laminators", ["Laminators"], aliases=["laminator", "laminating machine"]),
    _c(
        "laminating_pouches",
        "laminating pouches and sheets",
        ["Laminating Supplies"],
        aliases=["laminating pouch", "laminating sheet"],
    ),
    _c(
        "mouse_pads",
        "mouse pads and wrist rests",
        ["Mouse Pads & Wrist Rests"],
        aliases=["mouse pad", "mousepad", "wrist rest"],
    ),
    _c("clipboards", "clipboards", ["Clipboards"], aliases=["clipboard"]),
    _c(
        "desk_organizers",
        "desk organizers, pencil cups and letter trays",
        [
            "Desk Supplies Holders & Dispensers",
            "Letter Trays & Stacking Supports",
            "Drawer Organizers",
        ],
        aliases=[
            "desk organizer",
            "pencil holder",
            "pen holder",
            "pencil cup",
            "letter tray",
            "drawer organizer",
        ],
    ),
    _c(
        "business_card_holders",
        "business card holders",
        ["Business Card Holders"],
        aliases=["business card holder", "card holder"],
    ),
    _c(
        "paper_trimmers",
        "paper trimmers",
        ["Paper Trimmers & Blades"],
        aliases=["paper trimmer", "paper cutter"],
    ),
    _c(
        "rulers",
        "rulers and tape measures",
        ["Rulers & Tape Measures"],
        aliases=["ruler", "tape measure"],
    ),
    _c(
        "phones",
        "landline phones",
        ["Landline Phones"],
        aliases=["phone", "telephone", "cordless phone", "landline"],
    ),
    _c(
        "office_chairs",
        "office chairs",
        ["Chairs & Sofas"],
        aliases=["chair", "office chair", "desk chair"],
    ),
    _c("scanners", "scanners", ["Scanners"], aliases=["scanner", "document scanner"]),
    _c(
        "storage_boxes",
        "storage file boxes",
        ["Storage File Boxes"],
        aliases=["storage box", "file box", "banker box"],
    ),
    _c(
        "index_dividers",
        "index dividers",
        ["Index Dividers"],
        aliases=["divider", "index divider", "tab divider"],
    ),
    _c(
        "stamps",
        "stamps and stamp pads",
        ["Stamps", "Stamp Pads & Refills"],
        aliases=["stamp", "rubber stamp", "stamp pad", "ink pad"],
    ),
    _c(
        "monitor_stands",
        "monitor stands and arms",
        ["Monitor Arms & Stands"],
        aliases=["monitor stand", "monitor arm", "monitor riser"],
    ),
    _c(
        "postal_scales",
        "postal scales",
        ["Postal Scales"],
        aliases=["postal scale", "shipping scale", "postage scale"],
    ),
]

CATEGORIES: dict[str, Category] = {c.key: c for c in _ALL}
KEYS: tuple[str, ...] = tuple(CATEGORIES)


def in_category(item_categories: Iterable[str] | None, key: str) -> bool:
    """Whether an item with this category path belongs to category `key`."""
    cat = CATEGORIES[key]
    path = set(item_categories or ())
    return bool(path.intersection(cat.include)) and not path.intersection(cat.exclude)


_ALIASES: list[tuple[tuple[str, ...], int, str]] = [
    (tuple(text.stem(w) for w in text.words(alias)), order, cat.key)
    for order, cat in enumerate(_ALL)
    for alias in cat.aliases
]


def match_alias(query: str) -> tuple[str, str] | None:
    """(category key, matched phrase) for the first product phrase in `query`, if any.

    The earliest match wins ('toner for my laser printer' is toner); at the same position the
    longest ('printer ink' over 'printer'), then the category order above.
    """
    ws = [text.stem(w) for w in text.words(query)]
    best = None
    for alias, order, key in _ALIASES:
        n = len(alias)
        for i in range(len(ws) - n + 1):
            if tuple(ws[i : i + n]) == alias:
                cand = (i, -n, order, key, " ".join(alias))
                if best is None or cand < best:
                    best = cand
                break
    return (best[3], best[4]) if best else None
