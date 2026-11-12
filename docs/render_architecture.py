"""Writes docs/architecture.svg, the README's animated architecture diagram.

    python3 docs/render_architecture.py

Standard library only. The picture is static SVG plus CSS keyframes (GitHub shows an SVG as an
image, so no script can run in it). One 20-second loop follows three paths in turn:

  blue    a recommendation request through the api's five stages;
  teal    a click event through Redpanda to the consumers, the stores and the near-line worker;
  violet  the offline loop from the event log to a released model bundle.

Change the geometry or the timeline here and run the script again; do not edit the SVG.
"""

from __future__ import annotations

from pathlib import Path

W, H = 900, 616
LOOP_S = 20

BLUE, TEAL, VIOLET = "#0a7cff", "#0d9488", "#7c5cff"

style: list[str] = []
body: list[str] = []
overlay: list[str] = []  # comets, drawn last
_frames = 0


def window(start: float, end: float, fade: float = 0.6) -> str:
    """Keyframes for something that is visible from `start` to `end` (percent of the loop)."""
    global _frames
    _frames += 1
    name = f"w{_frames}"
    style.append(
        f"@keyframes {name} {{ 0%, {start:g}% {{ opacity: 0; }} {start + fade:g}%, {end:g}% {{ opacity: 1; }} "
        f"{end + fade:g}%, 100% {{ opacity: 0; }} }}"
    )
    return name


def comet(d: str, color: str, start: float, end: float) -> None:
    """A bright head with a soft tail that runs along path `d` between two points of the loop."""
    global _frames
    for cls, a, b in (("tail", 0.3, -1.0), ("head", 0.02, -1.28)):
        _frames += 1
        name = f"p{_frames}"
        style.append(
            f"@keyframes {name} {{ 0%, {start:g}% {{ opacity: 0; stroke-dashoffset: {a}; }} "
            f"{start + 0.01:g}% {{ opacity: 1; stroke-dashoffset: {a}; }} "
            f"{end:g}% {{ opacity: 1; stroke-dashoffset: {b}; }} "
            f"{end + 0.01:g}%, 100% {{ opacity: 0; stroke-dashoffset: {b}; }} }}"
        )
        overlay.append(f'<path d="{d}" pathLength="1" class="{cls}" stroke="{color}" style="animation-name:{name}"/>')


def wire(d: str, arrow: bool = True) -> None:
    end = ' marker-end="url(#arrow)"' if arrow else ""
    body.append(f'<path d="{d}" class="wire"{end}/>')


def card(x: float, y: float, w: float, h: float, cls: str = "card") -> None:
    body.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" class="{cls}"/>')


def ring(x: float, y: float, w: float, h: float, color: str, *spans: tuple[float, float], rx: int = 12) -> None:
    for start, end in spans:
        body.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" class="ring" stroke="{color}" '
            f'style="animation-name:{window(start, end)}"/>'
        )


def text(x: float, y: float, s: str, cls: str, anchor: str = "start") -> None:
    a = "" if anchor == "start" else f' text-anchor="{anchor}"'
    body.append(f'<text x="{x}" y="{y}" class="{cls}"{a}>{s}</text>')


def titled(x: float, y: float, w: float, h: float, eyebrow: str | None, name: str, *subs: str) -> None:
    card(x, y, w, h)
    ty = y + 18
    if eyebrow:
        text(x + 12, ty, eyebrow, "eyebrow")
        ty += 18
    else:
        ty += 6
    text(x + 12, ty, name, "name")
    for s in subs:
        ty += 16
        text(x + 12, ty, s, "sub")


# ---------------------------------------------------------------------------------- frame
body.append(f'<rect x="0.5" y="0.5" width="{W - 1}" height="{H - 1}" rx="16" class="bg"/>')
text(20, 28, "RECLOOP · SYSTEM ARCHITECTURE", "eyebrow")
for i, (label, color, span) in enumerate(
    [("request", BLUE, (0.5, 38.5)), ("events", TEAL, (39.5, 68.5)), ("offline loop", VIOLET, (69.5, 98.5))]
):
    x = 640 + i * 82
    body.append(f'<circle cx="{x}" cy="24" r="4" fill="{color}"/>')
    text(x + 9, 28, label, "legend")
    text(x + 9, 28, label, "legend on")
    body[-1] = body[-1].replace('class="legend on"', f'class="legend on" style="animation-name:{window(*span)}"')

# ---------------------------------------------------------------------------------- top row
STORE = (20, 44, 160, 92)
AGENT = (20, 152, 160, 84)
API = (216, 44, 344, 216)
MS = (596, 44, 284, 112)
WORKER = (596, 172, 284, 64)

titled(*STORE, "NEXT.JS 16", "Store", "feed · item pages · debug", "ad console · admin")
titled(*AGENT, "FASTAPI · CLAUDE", "Shopping assistant", "LLM parses, code checks", "price, stock, devices")

card(*API, cls="card strong")
text(228, 62, "GO · serving/", "eyebrow")
text(228, 80, "api", "title")
text(548, 80, "one 100 ms budget per request", "sub", "end")
STAGES = [
    ("User context", "history, purchases, seen", "5 ms", (4, 9)),
    ("Recall", "SASRec, ItemCF, Pop, gen", "35 ms", (9, 16)),
    ("Fuse + filter", "weighted RRF, stock, brand cap", "", (16, 20)),
    ("Rank", "LightGBM, 300 candidates", "25 ms", (20, 26)),
    ("Ads", "GSP auction, budget, caps", "10 ms", (26, 32)),
]
REQUEST_END = 38
stage_y = []
for i, (name, detail, budget, (start, end)) in enumerate(STAGES):
    y = 92 + 30 * i
    stage_y.append(y + 13)
    body.append(f'<rect x="228" y="{y}" width="320" height="26" rx="8" class="row"/>')
    body.append(
        f'<rect x="228" y="{y}" width="320" height="26" rx="8" class="rowon" style="animation-name:{window(start, end)}"/>'
    )
    body.append(f'<circle cx="243" cy="{y + 13}" r="8" class="badge"/>')
    body.append(
        f'<circle cx="243" cy="{y + 13}" r="8" class="lit" fill="{BLUE}" style="animation-name:{window(start, REQUEST_END)}"/>'
    )
    text(243, y + 17, str(i + 1), "badgenum", "middle")
    text(259, y + 17.5, name, "stage")
    text(352, y + 17.5, detail, "sub")
    if budget:
        text(540, y + 17.5, budget, "budget", "end")
text(388, 253, "gRPC deadlines · circuit breakers · degrade, never fail", "edge", "middle")

card(*MS)
text(608, 62, "PYTHON · gRPC", "eyebrow")
text(608, 80, "Model server", "name")
text(868, 80, "several bundle versions", "sub", "end")
for y, name, detail, span in ((90, "Retrieve", "SASRec (ONNX) + FAISS", (12.5, 16)), (120, "Rank", "LightGBM (ONNX) + pCTR", (23.5, 26.5))):
    body.append(f'<rect x="608" y="{y}" width="260" height="24" rx="8" class="row"/>')
    body.append(
        f'<rect x="608" y="{y}" width="260" height="24" rx="8" class="rowon" style="animation-name:{window(*span)}"/>'
    )
    text(620, y + 16.5, name, "stage")
    text(690, y + 16.5, detail, "sub")

titled(*WORKER, "MINIONEREC · NEAR-LINE", "Generative recall", "Qwen2.5-1.5B emits semantic IDs")

# ---------------------------------------------------------------------------------- stores
PG = (20, 286, 203, 60)
RP = (239, 286, 203, 60)
CACHE = (458, 286, 203, 60)
STATE = (677, 286, 203, 60)
titled(*PG, None, "PostgreSQL 18 + pgvector", "catalog, events, ledger, versions")
titled(*RP, None, "Redpanda", "event log, 6 partitions by user")
titled(*CACHE, None, "Redis · cache", "pages, candidate lists · LRU")
titled(*STATE, None, "Redis · state", "context, caps, spend · no eviction")

CONS = (239, 380, 422, 66)
card(*CONS)
text(251, 398, "STREAM CONSUMERS · each event applied once", "eyebrow")
for x, w, label in ((251, 128, "aggregator → state"), (387, 128, "log-sink → events"), (523, 126, "billing → ledger")):
    body.append(f'<rect x="{x}" y="408" width="{w}" height="26" rx="13" class="pill"/>')
    text(x + w / 2, 425, label, "pilltext", "middle")

# ---------------------------------------------------------------------------------- offline
SPARK = (20, 480, 203, 60)
TRAIN = (239, 480, 203, 60)
BUNDLE = (458, 480, 203, 60)
RELEASE = (677, 480, 203, 60)
titled(*SPARK, None, "Spark sample job", "attribution, delayed labels")
titled(*TRAIN, None, "Training", "SASRec · LightGBM · pCTR")
titled(*BUNDLE, None, "Versioned bundle", "ONNX, FAISS, golden queries")
titled(*RELEASE, None, "Release", "validate → canary 5% → active")

body.append('<rect x="20" y="562" width="860" height="34" rx="17" class="strip"/>')
text(36, 583.5, "RUNS ON", "eyebrow")
x = 100
for label, w in (
    ("Docker Compose", 106),
    ("Kubernetes · Helm", 118),
    ("HPA · KEDA", 80),
    ("Prometheus · Grafana", 136),
    ("OpenTelemetry", 98),
    ("k6 + simulator", 98),
    ("GitHub Actions", 100),
):
    body.append(f'<rect x="{x}" y="567" width="{w}" height="24" rx="12" class="pill"/>')
    text(x + w / 2, 583, label, "pilltext", "middle")
    x += w + 6

# ---------------------------------------------------------------------------------- wires
P_STORE_API = "M180 90 H216"
P_API_STORE = "M216 90 H180"
P_STORE_AGENT = "M100 136 V152"
P_AGENT_API = "M180 194 H216"
P_AGENT_PG = "M60 236 V286"
P_RETRIEVE = f"M560 {stage_y[1]} H578 V102 H608"
P_RANK = f"M560 {stage_y[3]} H586 V132 H608"
P_API_PG = "M240 260 V272 H121 V286"
P_API_RP = "M340 260 V286"
P_API_CACHE = "M520 260 V286"
P_API_STATE = "M548 260 V272 H720 V286"
P_WORKER_STATE = "M778 236 V286"
P_STATE_WORKER = "M778 286 V236"
P_RP_CONS = "M340 346 V380"
P_CONS_STATE = "M661 413 H778 V346"
P_CONS_PG = "M239 413 H121 V346"
P_PG_SPARK = "M60 346 V480"
P_SPARK_TRAIN = "M223 510 H239"
P_TRAIN_BUNDLE = "M442 510 H458"
P_BUNDLE_RELEASE = "M661 510 H677"
P_RELEASE_MS = "M880 510 H890 V100 H880"

for d in (
    P_STORE_API, P_STORE_AGENT, P_AGENT_API, P_AGENT_PG, P_RETRIEVE, P_RANK, P_API_PG, P_API_RP, P_API_CACHE,
    P_API_STATE, P_RP_CONS, P_CONS_STATE, P_CONS_PG, P_PG_SPARK, P_SPARK_TRAIN, P_TRAIN_BUNDLE,
    P_BUNDLE_RELEASE, P_RELEASE_MS,
):  # fmt: skip
    wire(d)
wire(P_WORKER_STATE, arrow=False)

# ---------------------------------------------------------------------------------- timeline
# Blue: one recommendation request.
comet(P_STORE_API, BLUE, 1, 4)
comet(P_API_STATE, BLUE, 5, 8)
ring(*STATE, BLUE, (7, 10), (29, 32))
comet(P_RETRIEVE, BLUE, 10, 13)
comet(P_RANK, BLUE, 21, 24)
comet(P_API_STATE, BLUE, 27, 30)
comet(P_API_STORE, BLUE, 32.5, 36)
ring(*STORE, BLUE, (0.5, 2.5), (35.5, 38.5))

# Teal: a click, from the api's event log to the consumers, the stores and the near-line worker.
ring(*STORE, TEAL, (39.5, 41.5))
comet(P_STORE_API, TEAL, 40, 43)
comet(P_API_RP, TEAL, 43.5, 47)
ring(*RP, TEAL, (46.5, 51))
comet(P_RP_CONS, TEAL, 48, 51)
ring(*CONS, TEAL, (50.5, 58))
comet(P_CONS_STATE, TEAL, 52, 57)
comet(P_CONS_PG, TEAL, 52, 57)
ring(*PG, TEAL, (56.5, 60))
ring(*STATE, TEAL, (56.5, 60), (66, 68.5))
comet(P_STATE_WORKER, TEAL, 59, 62)
ring(*WORKER, TEAL, (61.5, 66))
comet(P_WORKER_STATE, TEAL, 63.5, 66.5)

# Violet: from the event log to a model version that serves.
ring(*PG, VIOLET, (69.5, 71.5))
comet(P_PG_SPARK, VIOLET, 70, 74)
ring(*SPARK, VIOLET, (73.5, 77))
comet(P_SPARK_TRAIN, VIOLET, 75.5, 77.5)
ring(*TRAIN, VIOLET, (77, 80.5))
comet(P_TRAIN_BUNDLE, VIOLET, 79, 81)
ring(*BUNDLE, VIOLET, (80.5, 84))
comet(P_BUNDLE_RELEASE, VIOLET, 82.5, 84.5)
ring(*RELEASE, VIOLET, (84, 89))
comet(P_RELEASE_MS, VIOLET, 86, 93)
ring(*MS, VIOLET, (92.5, 98))

# ---------------------------------------------------------------------------------- output
DESC = (
    "System architecture of RecLoop. The store (Next.js) and the shopping assistant (FastAPI, Claude) call the Go "
    "api. A recommendation request runs five stages inside one 100 ms budget: user context from Redis; recall from "
    "SASRec through the gRPC model server, ItemCF, popularity and generated items; weighted RRF fusion and filters; "
    "LightGBM ranking through the model server; and a second-price ad auction with budgets and frequency caps. The "
    "api reads and writes PostgreSQL with pgvector, a Redis instance for state and one for cache, and publishes "
    "events to Redpanda. Stream consumers apply each event once: the aggregator updates the user state, the log "
    "sink stores events in PostgreSQL and billing writes the charge ledger. A near-line worker runs MiniOneRec and "
    "writes generated items to Redis. Offline, a Spark job joins training samples from the event log, training "
    "produces a versioned bundle, and a release goes through validation and a 5 percent canary before the model "
    "server serves it. It runs on Docker Compose and on Kubernetes with Helm, HPA and KEDA, observed with "
    "Prometheus, Grafana and OpenTelemetry. A pulse follows a request, then a click event, then the offline loop."
)

CSS = f"""
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Helvetica, Arial, sans-serif; }}
    .eyebrow {{ font-size: 10.5px; font-weight: 600; letter-spacing: 0.08em; fill: #8e8e93; }}
    .title {{ font-size: 16px; font-weight: 650; fill: #1d1d1f; }}
    .name {{ font-size: 13.5px; font-weight: 600; fill: #1d1d1f; }}
    .stage {{ font-size: 12.5px; font-weight: 600; fill: #1d1d1f; }}
    .sub {{ font-size: 11.5px; fill: #6e6e73; }}
    .budget {{ font-size: 11px; font-weight: 600; fill: #8e8e93; }}
    .edge {{ font-size: 11px; fill: #8e8e93; }}
    .pilltext {{ font-size: 11.5px; font-weight: 600; fill: #6e6e73; }}
    .badgenum {{ font-size: 11px; font-weight: 700; fill: #ffffff; }}
    .legend {{ font-size: 11.5px; fill: #aeaeb2; }}
    .legend.on {{ fill: #1d1d1f; opacity: 0; animation: {LOOP_S}s linear infinite; }}

    .bg {{ fill: #fbfbfd; stroke: #e5e5ea; }}
    .card {{ fill: #ffffff; stroke: #e5e5ea; stroke-width: 1; }}
    .strong {{ stroke: #d1d1d6; }}
    .row {{ fill: #f5f5f7; }}
    .rowon {{ fill: #e8f0ff; stroke: {BLUE}; stroke-width: 1.5; opacity: 0; animation: {LOOP_S}s linear infinite; }}
    .badge {{ fill: #c7c7cc; }}
    .lit {{ opacity: 0; animation: {LOOP_S}s linear infinite; }}
    .ring {{ fill: none; stroke-width: 2; opacity: 0; animation: {LOOP_S}s linear infinite; }}
    .strip {{ fill: #f5f5f7; }}
    .pill {{ fill: #ffffff; stroke: #e5e5ea; }}
    .wire {{ fill: none; stroke: #d1d1d6; stroke-width: 1.5; }}

    /* A comet: a soft tail with a bright head. */
    .tail, .head {{ fill: none; stroke-linecap: round; opacity: 0; animation: {LOOP_S}s linear infinite; }}
    .tail {{ stroke-width: 3; stroke-opacity: 0.35; stroke-dasharray: 0.3 3; stroke-dashoffset: 0.3; }}
    .head {{ stroke-width: 5; stroke-dasharray: 0.02 3; stroke-dashoffset: 0.02; }}
    @media (prefers-reduced-motion: reduce) {{
      .tail, .head, .ring, .lit, .rowon, .legend.on {{ animation: none; }}
    }}
"""

svg = "\n".join(
    [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" role="img" '
        'aria-labelledby="title desc">',
        '  <title id="title">RecLoop system architecture</title>',
        f'  <desc id="desc">{DESC}</desc>',
        "  <style>" + CSS + "    " + "\n    ".join(style) + "\n  </style>",
        '  <defs><marker id="arrow" viewBox="0 0 6 6" markerWidth="6" markerHeight="6" refX="5" refY="3" '
        'orient="auto" markerUnits="userSpaceOnUse"><path d="M0 0 L6 3 L0 6 z" fill="#c7c7cc"/></marker></defs>',
        *("  " + line for line in body),
        *("  " + line for line in overlay),
        "</svg>",
        "",
    ]
)
out = Path(__file__).with_name("architecture.svg")
out.write_text(svg, encoding="utf-8")
print(f"{out} ({len(svg):,} bytes)")
