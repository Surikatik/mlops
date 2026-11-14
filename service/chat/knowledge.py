import csv
import json
import re
from pathlib import Path
from statistics import mean, median

from haystack import Document

MARKDOWN_FILES = ("readme.md", "disign_doc.md")
FEATURES_FILE = "features/all_features.csv"
METRICS_FILE = "reports/metrics.json"
ROOM_NAMES = {0: "студия", 1: "1-комнатная", 2: "2-комнатная", 3: "3-комнатная"}
ROOM_WORDS = {1: "однокомнатные", 2: "двухкомнатные", 3: "трёхкомнатные", 4: "четырёхкомнатные", 5: "пятикомнатные"}
HEADING = re.compile(r"^(#{1,4})\s+(.+?)\s*$")
MIN_SECTION_LENGTH = 40
MAX_CHUNK_LENGTH = 700


def rub(value):
    return f"{int(round(value)):,}".replace(",", " ") + " руб."


def room_name(rooms):
    return ROOM_NAMES.get(rooms, f"{rooms}-комнатная")


def split_chunks(text, limit=MAX_CHUNK_LENGTH):
    chunks = []
    current = ""
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block:
            continue
        if current and len(current) + len(block) > limit:
            chunks.append(current)
            current = block
        else:
            current = f"{current}\n\n{block}" if current else block
    if current:
        chunks.append(current)
    return chunks


def markdown_sections(path):
    path = Path(path)
    documents = []
    titles = {}
    title = path.stem
    lines = []

    def flush():
        content = "\n".join(lines).strip()
        if len(content) < MIN_SECTION_LENGTH:
            return
        for chunk in split_chunks(content):
            documents.append(
                Document(
                    content=f"{title}\n{chunk}",
                    meta={"source": path.name, "title": title, "kind": "docs"},
                )
            )

    in_code = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            in_code = not in_code
        match = None if in_code else HEADING.match(line)
        if not match:
            lines.append(line)
            continue
        flush()
        level = len(match.group(1))
        titles = {k: v for k, v in titles.items() if k < level}
        titles[level] = match.group(2)
        levels = [k for k in sorted(titles) if k > 1] or sorted(titles)
        title = " / ".join(titles[k] for k in levels)
        lines = []
    flush()
    return documents


def read_flats(path):
    flats = []
    with open(path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                flats.append(
                    {
                        "id": row["id"],
                        "rooms": int(float(row["rooms"])),
                        "area": float(row["area"]),
                        "floor": int(float(row["floor"])),
                        "total_floors": int(float(row["total_floors"])),
                        "price": float(row["price"]),
                    }
                )
            except (KeyError, ValueError):
                continue
    return flats


def listing_documents(flats):
    documents = []
    for flat in flats:
        content = (
            f"Объявление {flat['id']}: {room_name(flat['rooms'])} квартира, "
            f"площадь {flat['area']:g} м², этаж {flat['floor']} из {flat['total_floors']}, "
            f"цена {rub(flat['price'])}, цена за метр {rub(flat['price'] / flat['area'])}"
        )
        documents.append(
            Document(
                content=content,
                meta={
                    "source": FEATURES_FILE,
                    "title": f"объявление {flat['id']}",
                    "kind": "listing",
                },
            )
        )
    return documents


def describe_group(name, flats):
    prices = [flat["price"] for flat in flats]
    areas = [flat["area"] for flat in flats]
    per_meter = [flat["price"] / flat["area"] for flat in flats]
    cheapest = min(flats, key=lambda flat: flat["price"])
    priciest = max(flats, key=lambda flat: flat["price"])
    return (
        f"Статистика по объявлениям, {name}: количество {len(flats)}, "
        f"медианная цена {rub(median(prices))}, средняя цена {rub(mean(prices))}, "
        f"минимальная цена {rub(cheapest['price'])} (объявление {cheapest['id']}, {cheapest['area']:g} м²), "
        f"максимальная цена {rub(priciest['price'])} (объявление {priciest['id']}, {priciest['area']:g} м²), "
        f"средняя площадь {mean(areas):.1f} м², средняя цена за метр {rub(mean(per_meter))}"
    )


def stats_documents(flats):
    if not flats:
        return []
    groups = {"все квартиры": flats}
    for rooms in sorted({flat["rooms"] for flat in flats}):
        name = "студии" if rooms == 0 else f"{room_name(rooms)} квартиры"
        if rooms in ROOM_WORDS:
            name = f"{ROOM_WORDS[rooms]} ({name})"
        groups[name] = [flat for flat in flats if flat["rooms"] == rooms]
    return [
        Document(
            content=describe_group(name, group),
            meta={"source": FEATURES_FILE, "title": f"статистика: {name}", "kind": "stats"},
        )
        for name, group in groups.items()
    ]


def metrics_documents(path):
    with open(path, encoding="utf-8") as f:
        report = json.load(f)
    lines = [f"Результаты обучения моделей. Лучшая модель: {report['best_model']}."]
    for name, metrics in report.items():
        if not isinstance(metrics, dict):
            continue
        lines.append(
            f"Модель {name}: MAE на кросс-валидации {rub(metrics['val_mae'])}, "
            f"MAE на отложенной выборке {rub(metrics['holdout_mae'])}, "
            f"MAPE на отложенной выборке {metrics['holdout_mape'] * 100:.1f}%"
        )
    return [
        Document(
            content="\n".join(lines),
            meta={"source": METRICS_FILE, "title": "метрики моделей", "kind": "metrics"},
        )
    ]


def build_documents(root="."):
    root = Path(root)
    documents = []
    for name in MARKDOWN_FILES:
        if (root / name).exists():
            documents.extend(markdown_sections(root / name))
    if (root / FEATURES_FILE).exists():
        flats = read_flats(root / FEATURES_FILE)
        documents.extend(stats_documents(flats))
        documents.extend(listing_documents(flats))
    if (root / METRICS_FILE).exists():
        documents.extend(metrics_documents(root / METRICS_FILE))
    return documents
