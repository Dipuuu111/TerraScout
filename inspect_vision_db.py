"""Inspect and visually audit TerraScout observations stored by vision_train.py.

Examples (PowerShell):
    python inspect_vision_db.py --database data/terrascout.db --list
    python inspect_vision_db.py --database data/terrascout.db --show survey_0042
    python inspect_vision_db.py --database data/terrascout.db --annotate survey_0042

The --annotate command requires Pillow: python -m pip install Pillow
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


def connect_db(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        raise FileNotFoundError(
            f"Database not found: {path}\n"
            "Check --database and run vision_train.py at least once."
        )
    connection = sqlite3.connect(str(path))
    connection.row_factory = sqlite3.Row
    return connection


def decode_json(value: str | None, fallback: Any) -> Any:
    if not value:
        return fallback
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return fallback


def human_label(value: Any) -> str:
    """Render machine-readable enum values as a compact human-readable label."""
    if not isinstance(value, str) or not value:
        return "Unknown"
    # Keep legacy values readable while displaying the expanded current taxonomy naturally.
    legacy = {
        "moderate_to_high": "Moderate to High (legacy rating)",
        "low": "Low (legacy rating)",
    }
    if value in legacy:
        return legacy[value]
    return value.replace("_", " ").capitalize()


def find_rows(connection: sqlite3.Connection, identifier: str) -> list[sqlite3.Row]:
    # Match exact photo_id first, then source-path/filename substrings for convenience.
    exact = connection.execute(
        "SELECT * FROM observations WHERE photo_id = ? ORDER BY analyzed_at_utc DESC",
        (identifier,),
    ).fetchall()
    if exact:
        return list(exact)
    partial = f"%{identifier}%"
    return list(connection.execute(
        "SELECT * FROM observations WHERE source_path LIKE ? ORDER BY analyzed_at_utc DESC",
        (partial,),
    ).fetchall())


def print_list(connection: sqlite3.Connection, limit: int | None) -> int:
    sql = "SELECT * FROM observations ORDER BY analyzed_at_utc DESC"
    params: tuple[Any, ...] = ()
    if limit is not None:
        sql += " LIMIT ?"
        params = (limit,)
    rows = connection.execute(sql, params).fetchall()
    if not rows:
        print("The database exists, but it contains no observations yet.")
        return 0

    print(f"Observations: {len(rows)}\n")
    for i, row in enumerate(rows, 1):
        terrain = decode_json(row["terrain_json"], {})
        observed_objects = decode_json(row["candidate_objects_json"], [])
        uncertainties = decode_json(row["uncertainties_json"], [])
        filename = Path(row["source_path"]).name
        materials = ", ".join(terrain.get("surface_materials", [])) or "(none recorded)"
        abundance = terrain.get("relative_rock_abundance", "unknown")
        traversal = terrain.get("visual_traversal_difficulty", "unknown")
        slope = terrain.get("slope", {})
        slope_assessment = slope.get("assessment", "unknown") if isinstance(slope, dict) else "unknown"
        slope_direction = slope.get("direction", "unknown") if isinstance(slope, dict) else "unknown"
        drop_off = terrain.get("drop_off", {})
        drop_off_assessment = drop_off.get("assessment", "unknown") if isinstance(drop_off, dict) else "unknown"
        print(f"[{i}] photo_id: {row['photo_id']}")
        print(f"    file: {filename}")
        print(f"    analyzed_utc: {row['analyzed_at_utc']} | model: {row['model']}")
        print(f"    materials: {materials}")
        print(
            f"    surface profile: {human_label(terrain.get('surface_profile', 'unknown'))} "
            f"| slope: {human_label(slope_assessment)} ({human_label(slope_direction)}) "
            f"| drop-off: {human_label(drop_off_assessment)} "
            f"| visual traversal grade: {human_label(traversal)}"
        )
        distribution = terrain.get("obstacle_distribution", "unknown")
        print(
            f"    rock abundance: {abundance} | roughness: {terrain.get('apparent_roughness', 'unknown')} "
            f"| obstacles: {distribution}"
        )
        reason = terrain.get("visual_traversal_reason", "")
        if reason:
            print(f"    traversal basis: {reason}")
        print(f"    observed objects: {len(observed_objects)} | uncertainties: {len(uncertainties)}")
        if row["latitude"] is not None and row["longitude"] is not None:
            print(f"    coordinates: {row['latitude']:.7f}, {row['longitude']:.7f}")
        print()
    return 0


def print_record(row: sqlite3.Row) -> None:
    print(f"photo_id: {row['photo_id']}")
    print(f"source_path: {row['source_path']}")
    print(f"analyzed_at_utc: {row['analyzed_at_utc']}")
    print(f"image_modified_at_utc: {row['image_modified_at_utc']}")
    print(f"model: {row['model']}")
    print(f"image_sha256: {row['image_sha256']}")
    print(f"latitude: {row['latitude']} | longitude: {row['longitude']}")
    payload = decode_json(row["payload_json"], {"error": "payload_json is not valid JSON"})
    terrain = payload.get("terrain", {}) if isinstance(payload, dict) else {}
    slope = terrain.get("slope", {}) if isinstance(terrain, dict) else {}
    drop_off = terrain.get("drop_off", {}) if isinstance(terrain, dict) else {}
    print("\nTerrain classification summary:")
    print(f"  Surface profile: {human_label(terrain.get('surface_profile', 'unknown'))}")
    print(
        f"  Slope: {human_label(slope.get('assessment', 'unknown') if isinstance(slope, dict) else 'unknown')} "
        f"({human_label(slope.get('direction', 'unknown') if isinstance(slope, dict) else 'unknown')})"
    )
    print(f"  Drop-off: {human_label(drop_off.get('assessment', 'unknown') if isinstance(drop_off, dict) else 'unknown')}")
    if isinstance(drop_off, dict) and drop_off.get("reason"):
        print(f"  Drop-off basis: {drop_off['reason']}")
    if isinstance(slope, dict) and slope.get("reason"):
        print(f"  Slope basis: {slope['reason']}")
    print(f"  Traversal grade: {human_label(terrain.get('visual_traversal_difficulty', 'unknown'))}")
    print(f"  Reason: {terrain.get('visual_traversal_reason', '(not stored)')}")
    print("\nStored model observation (payload_json):")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def annotate_row(row: sqlite3.Row, output: Path | None) -> int:
    source = Path(row["source_path"])
    if not source.is_file():
        raise FileNotFoundError(
            f"Original image is no longer at the stored path: {source}\n"
            "Move/copy the image back there or update the source_path in the database."
        )
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError as exc:
        raise RuntimeError(
            "Image annotation needs Pillow. Install it with: python -m pip install Pillow"
        ) from exc

    observed_objects = decode_json(row["candidate_objects_json"], [])
    image = Image.open(source).convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    line_width = max(2, round(min(width, height) * 0.004))
    try:
        font = ImageFont.truetype("arial.ttf", max(12, round(min(width, height) * 0.025)))
    except OSError:
        font = ImageFont.load_default()

    boxes_drawn = 0
    for candidate in observed_objects:
        box = candidate.get("bounding_box_normalized")
        if not isinstance(box, list) or len(box) != 4:
            continue
        x0 = max(0, min(width - 1, round(float(box[0]) * width)))
        y0 = max(0, min(height - 1, round(float(box[1]) * height)))
        x1 = max(0, min(width - 1, round(float(box[2]) * width)))
        y1 = max(0, min(height - 1, round(float(box[3]) * height)))
        if x1 <= x0 or y1 <= y0:
            continue
        draw.rectangle((x0, y0, x1, y1), outline=(255, 55, 55), width=line_width)
        label = f"{candidate.get('label', 'object')} ({float(candidate.get('confidence', 0.0)):.2f})"
        bbox = draw.textbbox((0, 0), label, font=font)
        text_w, text_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        label_y = max(0, y0 - text_h - 8)
        draw.rectangle((x0, label_y, min(width - 1, x0 + text_w + 8), label_y + text_h + 6), fill=(255, 55, 55))
        draw.text((x0 + 4, label_y + 3), label, fill=(255, 255, 255), font=font)
        boxes_drawn += 1

    if boxes_drawn == 0:
        print("No valid observed-object bounding boxes were stored for this image.")
        print("That does not necessarily mean there are no rocks: terrain-level surface observations are stored separately.")

    destination = output or (Path("data") / "vision_audit" / f"{row['photo_id']}_annotated.png")
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)
    print(f"Annotated image saved: {destination.resolve()}")
    print(f"Bounding boxes drawn: {boxes_drawn} / {len(observed_objects)} observed object(s)")
    print("Compare each box with the original: does it enclose the reported rock/object, without pointing at unrelated terrain?")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect TerraScout's SQLite vision survey database.")
    parser.add_argument("--database", type=Path, default=Path("data/terrascout.db"), help="SQLite database path")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--list", action="store_true", help="List stored observations")
    mode.add_argument("--show", metavar="PHOTO_ID_OR_PATH_PART", help="Print full stored JSON for a photo")
    mode.add_argument("--annotate", metavar="PHOTO_ID_OR_PATH_PART", help="Draw stored observed-object boxes on the original image")
    parser.add_argument("--limit", type=int, help="Maximum observations printed with --list")
    parser.add_argument("--output", type=Path, help="Output path for --annotate")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.limit is not None and args.limit < 1:
        print("--limit must be at least 1", file=sys.stderr)
        return 2
    if args.output is not None and not args.annotate:
        print("--output can only be used with --annotate", file=sys.stderr)
        return 2
    try:
        connection = connect_db(args.database.expanduser().resolve())
        try:
            if args.list:
                return print_list(connection, args.limit)
            rows = find_rows(connection, args.show or args.annotate)
            if not rows:
                print(f"No observation found for: {args.show or args.annotate}", file=sys.stderr)
                return 1
            if args.show:
                for index, row in enumerate(rows, 1):
                    if len(rows) > 1:
                        print(f"===== Match {index}/{len(rows)} =====")
                    print_record(row)
                return 0
            if len(rows) > 1:
                print("More than one image matched. Use a more specific photo ID or filename substring:")
                for row in rows:
                    print(f"  {row['photo_id']}  |  {row['source_path']}")
                return 2
            return annotate_row(rows[0], args.output)
        finally:
            connection.close()
    except (OSError, sqlite3.Error, RuntimeError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
