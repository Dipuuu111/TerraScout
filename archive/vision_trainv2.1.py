"""TerraScout vision survey pipeline.

This is an image-survey/inference module, not a neural-network training loop.
It sends each image to GLM-5.3-Flash through Nebius Token Factory, validates
terrain metadata and a mission-neutral inventory of visible objects, then stores
the result in SQLite for later mission-specific reasoning.

Environment variables:
    NEBIUS_API_KEY  Required: Nebius Token Factory API key.
    GLM_MODEL       Optional: model identifier (default below).
    NEBIUS_BASE_URL Optional: API base URL (default below).

Example:
    python vision_train.py --input data/survey_images --recursive
    python vision_train.py --input data/survey_images/frame_0042.jpg
    python vision_train.py --input data/survey_images --database data/terrascout.db --limit 10

Structured output uses Nebius Token Factory JSON mode plus guided_json schema constraints.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # .env loading is optional; environment variables still work without python-dotenv.
    pass

try:
    from openai import OpenAI
    from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
except ImportError as exc:  # pragma: no cover - exercised only in an incomplete environment
    raise SystemExit(
        "Missing dependency: openai. Install project dependencies with "
        "`python -m pip install -r requirements.txt`."
    ) from exc

LOGGER = logging.getLogger("terrascout.vision")

DEFAULT_MODEL = "zai-org/GLM-5.3-Flash"
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

ENUMS: dict[str, set[str]] = {
    "relative_rock_abundance": {"low", "moderate", "high", "unknown"},
    "apparent_roughness": {"low", "moderate", "high", "unknown"},
    "obstacle_distribution": {"isolated", "scattered", "widespread", "continuous", "unknown"},
    "surface_profile": {
        "smooth", "mostly_smooth", "gently_uneven", "undulating",
        "uneven", "broken_rocky", "mixed", "unknown"
    },
    "visual_traversal_difficulty": {
        "very_easy", "easy", "easy_to_moderate", "moderate",
        "moderate_to_difficult", "difficult", "very_difficult", "unknown"
    },
    "slope_assessment": {
        "flat", "nearly_flat", "gentle", "moderate", "steep", "very_steep", "unknown"
    },
}

GUIDED_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "photo_id": {"type": "string"},
        "terrain": {
            "type": "object",
            "properties": {
                "surface_materials": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                "dominant_colors": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
                "secondary_colors": {"type": "array", "maxItems": 3, "items": {"type": "string"}},
                "rock_shapes": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
                "surface_profile": {"type": "string", "enum": ["smooth", "mostly_smooth", "gently_uneven", "undulating", "uneven", "broken_rocky", "mixed", "unknown"]},
                "relative_rock_abundance": {"type": "string", "enum": ["low", "moderate", "high", "unknown"]},
                "apparent_roughness": {"type": "string", "enum": ["low", "moderate", "high", "unknown"]},
                "obstacle_distribution": {"type": "string", "enum": ["isolated", "scattered", "widespread", "continuous", "unknown"]},
                "visual_traversal_difficulty": {"type": "string", "enum": ["very_easy", "easy", "easy_to_moderate", "moderate", "moderate_to_difficult", "difficult", "very_difficult", "unknown"]},
                "visual_traversal_reason": {"type": "string"},
                "slope": {
                    "type": "object",
                    "properties": {
                        "assessment": {"type": "string", "enum": ["flat", "nearly_flat", "gentle", "moderate", "steep", "very_steep", "unknown"]},
                        "reason": {"type": "string"},
                    },
                    "required": ["assessment", "reason"],
                    "additionalProperties": False,
                },
            },
            "required": [
                "surface_materials", "dominant_colors", "secondary_colors", "rock_shapes",
                "surface_profile", "relative_rock_abundance", "apparent_roughness", "obstacle_distribution",
                "visual_traversal_difficulty", "visual_traversal_reason", "slope"
            ],
            "additionalProperties": False,
        },
        "observed_objects": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "description": {"type": "string"},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "bounding_box_normalized": {
                        "anyOf": [
                            {
                                "type": "array", "minItems": 4, "maxItems": 4,
                                "items": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["label", "description", "confidence", "bounding_box_normalized"],
                "additionalProperties": False,
            },
        },
        "uncertainties": {"type": "array", "maxItems": 4, "items": {"type": "string"}},
    },
    "required": ["photo_id", "terrain", "observed_objects", "uncertainties"],
    "additionalProperties": False,
}


SYSTEM_PROMPT = """You are TerraScout's visual survey analyst for a mobile exploration rover.
Your job at this stage is neutral perception and scene description, NOT target
selection, object ranking, mission planning, or a movement decision. The operator
will specify what matters for a mission later. Record what is visibly present
without assuming which object the operator will want.

Analyze only evidence visible in the supplied photograph. Do not invent dimensions,
coordinates, object identities, mineral composition, slope, traction, or hidden
geometry. A single image usually cannot establish real-world scale or safe rover
clearance. Mark unknowns explicitly.

IMPORTANT: assess overall terrain separately from individual objects. The terrain
ratings must describe the predominant visible ground / continuous traversable-looking
surface across the scene, not the single worst object in the photograph. Do NOT
automatically assign moderate-or-higher roughness or traversal difficulty merely
because one or a few large rocks are visible. If most visible ground is smooth sand,
soil, compacted earth, or relatively even fine gravel and larger rocks are isolated,
describe the surface based on that majority and record the isolated rocks separately
in observed_objects. Mention uncertain clearance as an uncertainty rather than
penalizing the entire terrain rating without evidence.

Use these assessment rules:
- surface_materials: list materials in approximate order of how much visible ground
  they occupy, not by visual importance. These are qualitative image impressions.
- relative_rock_abundance: assess the apparent fraction/distribution of visible
  ground covered by rocks, not the size or count of the single largest rock. Use low
  for a few isolated rocks over mostly non-rock ground; moderate for noticeable
  but not dominant rock coverage; high when rocks cover a large/widespread part of
  the visible ground; unknown if the visible ground is insufficient.
- obstacle_distribution: isolated = one/few separated obstacles; scattered = several
  separated obstacles across the scene; widespread = obstacles/irregularities across
  much of the scene; continuous = nearly continuous rough/obstructed ground; unknown
  when not enough is visible.
- apparent_roughness: rate the typical visible ground surface across the scene, not
  the maximum roughness of a single object. Low when most ground looks even; moderate
  when widespread small irregularities are visible; high when most ground is visibly
  uneven, jagged, or broken; unknown if occlusion or perspective prevents assessment.
- surface_profile: describe the geometry/texture of the dominant ground separately
  from slope. Choose smooth for fairly even continuous ground; mostly_smooth when
  only minor irregularities occur; gently_uneven for recurring small surface changes;
  undulating for broad rolling highs/lows; uneven for frequent irregular relief;
  broken_rocky for predominantly fractured/jagged rock; mixed when no one profile
  dominates; unknown when not enough ground is visible.
- visual_traversal_difficulty: use this eight-level visual scale, not just "moderate":
  very_easy = continuous-looking, smooth ground with almost no visible interruptions;
  easy = mostly even ground with minor texture or a few small isolated obstacles;
  easy_to_moderate = generally even ground with intermittent pebbles, shallow ruts,
  or isolated rocks, with most of the visible surface still relatively continuous;
  moderate = repeated unevenness, coarse loose ground, or obstacles interrupting
  substantial portions of the visible scene;
  moderate_to_difficult = clearly rough/loose ground or many obstacles across much
  of the visible surface, with limited visually clear patches;
  difficult = dominant severe unevenness, deep-looking loose material, steep visible
  slope, or frequent substantial obstacles;
  very_difficult = extreme-looking continuous broken terrain, very steep visible
  ground, or dense major obstacles across most of the scene;
  unknown = ground/path cannot be assessed from this image. These are visual grades,
  not rover-clearance or safety guarantees. Do not raise the whole-scene grade just
  because one large rock is present; judge the dominant area and obstacle frequency.
- visual_traversal_reason: one concise sentence citing the dominant surface, surface
  profile and obstacle distribution. Explain the rating in evidence-based language.
- slope: use flat for no clear incline, nearly_flat for a slight but perceptible
  incline, gentle/moderate/steep/very_steep for progressively more pronounced visible
  slope. Assess only the ground plane; a tall rock is not a slope. Use unknown when
  perspective makes slope ambiguous. Slope and surface difficulty are separate fields:
  flat terrain may still be rough, and sloped terrain may have a smooth surface.

Return exactly one valid JSON object, no Markdown fences and no extra prose, using
this schema:
{
  "photo_id": "copy the provided photo_id exactly",
  "terrain": {
    "surface_materials": ["short labels, roughly ordered by visible coverage"],
    "dominant_colors": ["lowercase color labels"],
    "secondary_colors": ["lowercase color labels"],
    "rock_shapes": ["shapes actually visible, across sizes"],
    "surface_profile": "smooth|mostly_smooth|gently_uneven|undulating|uneven|broken_rocky|mixed|unknown",
    "relative_rock_abundance": "low|moderate|high|unknown",
    "apparent_roughness": "low|moderate|high|unknown",
    "obstacle_distribution": "isolated|scattered|widespread|continuous|unknown",
    "visual_traversal_difficulty": "very_easy|easy|easy_to_moderate|moderate|moderate_to_difficult|difficult|very_difficult|unknown",
    "visual_traversal_reason": "short, evidence-based explanation",
    "slope": {
      "assessment": "flat|nearly_flat|gentle|moderate|steep|very_steep|unknown",
      "reason": "brief visual evidence or why it is unknown"
    }
  },
  "observed_objects": [
    {
      "label": "concise label for the visible object or distinct cluster",
      "description": "observable traits only; no unsupported material/mineral claims",
      "confidence": 0.0,
      "bounding_box_normalized": [0.0, 0.0, 1.0, 1.0]
    }
  ],
  "uncertainties": ["specific ambiguity or missing evidence"]
}

Observed objects are a neutral scene inventory, not mission candidates or a ranking.
Record distinct visible objects/features of any apparent size when boundaries can be
reasonably distinguished. Do NOT prioritize large rocks just because they are large,
and do NOT omit a distinct smaller object solely because it is small. Include rocks
of different sizes, visible clusters, and other clearly identifiable ground objects
when useful. For repeated tiny pebbles whose individual boundaries are indistinct,
record the pebble/gravel patch as a cluster or describe it only at terrain level;
do not invent individual bounding boxes. Do not return every pebble. Up to 8 objects
may be recorded; spread coverage across distinct visible objects/regions rather than
ranking by size, presumed importance, or a future mission target. If more than 8 are
visible, describe representative objects and summarize the remainder as terrain.
When apparent relative size is useful, say small/medium/large relative to this image;
do not invent physical dimensions.

Bounding boxes are [x_min, y_min, x_max, y_max], normalized to 0..1 relative to image
width/height. Use null when the object's extent cannot be localized reliably. Confidence
is a subjective model estimate, not a calibrated probability. Use empty lists rather
than guessing. Include uncertainties when image quality, lighting, occlusion,
perspective, lack of scale, or unknown rover clearance limits a conclusion.
Keep descriptions concise. JSON must be complete and syntactically valid.
"""


def utc_now() -> str:
    """Return a timezone-aware ISO-8601 UTC timestamp."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def safe_photo_id(path: Path) -> str:
    """Create a readable ID from a filename stem without disclosing directory paths."""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("._-")
    return (stem or "survey_image")[:120]


def image_mime_type(path: Path) -> str:
    mime, _ = mimetypes.guess_type(path.name)
    if mime not in {"image/jpeg", "image/png", "image/webp", "image/bmp"}:
        raise ValueError(f"Unsupported or unrecognized image type: {path.name}")
    return mime


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating surrounding whitespace or a Markdown fence."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    decoder = json.JSONDecoder()
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Model response did not contain a JSON object.")
    value, _ = decoder.raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("Model response JSON root must be an object.")
    return value


def _string_list(value: Any, field_name: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"`{field_name}` must be a JSON array.")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"Every item in `{field_name}` must be a string.")
    # Remove blank strings and duplicates while preserving order.
    return list(dict.fromkeys(item.strip() for item in value if item.strip()))


def _enum_value(value: Any, allowed: set[str], field_name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"`{field_name}` must be a string.")
    normalized = value.strip().lower()
    if normalized not in allowed:
        raise ValueError(
            f"`{field_name}` has unsupported value {value!r}; expected one of "
            f"{', '.join(sorted(allowed))}."
        )
    return normalized


def _normalize_box(value: Any) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 4:
        raise ValueError("`bounding_box_normalized` must be null or a four-number array.")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("Bounding-box coordinates must be numbers.")
    box = [float(item) for item in value]
    if any(item < 0.0 or item > 1.0 for item in box):
        raise ValueError("Bounding-box coordinates must be between 0 and 1.")
    x_min, y_min, x_max, y_max = box
    if x_min > x_max or y_min > y_max:
        raise ValueError("Bounding-box minima must not exceed maxima.")
    return box


def validate_observation(payload: dict[str, Any], expected_photo_id: str) -> dict[str, Any]:
    """Validate and normalize model output before it is written to the database."""
    terrain_raw = payload.get("terrain")
    if not isinstance(terrain_raw, dict):
        raise ValueError("Model output is missing the `terrain` object.")

    slope_raw = terrain_raw.get("slope")
    if not isinstance(slope_raw, dict):
        raise ValueError("`terrain.slope` must be a JSON object.")
    slope_reason = slope_raw.get("reason", "")
    if not isinstance(slope_reason, str):
        raise ValueError("`terrain.slope.reason` must be a string.")

    traversal_reason = terrain_raw.get("visual_traversal_reason", "")
    if not isinstance(traversal_reason, str):
        raise ValueError("`terrain.visual_traversal_reason` must be a string.")

    terrain = {
        "surface_materials": _string_list(terrain_raw.get("surface_materials", []), "surface_materials"),
        "dominant_colors": _string_list(terrain_raw.get("dominant_colors", []), "dominant_colors"),
        "secondary_colors": _string_list(terrain_raw.get("secondary_colors", []), "secondary_colors"),
        "rock_shapes": _string_list(terrain_raw.get("rock_shapes", []), "rock_shapes"),
        "surface_profile": _enum_value(
            terrain_raw.get("surface_profile", "unknown"), ENUMS["surface_profile"], "surface_profile"
        ),
        "relative_rock_abundance": _enum_value(
            terrain_raw.get("relative_rock_abundance", "unknown"),
            ENUMS["relative_rock_abundance"], "relative_rock_abundance"
        ),
        "apparent_roughness": _enum_value(
            terrain_raw.get("apparent_roughness", "unknown"),
            ENUMS["apparent_roughness"], "apparent_roughness"
        ),
        "obstacle_distribution": _enum_value(
            terrain_raw.get("obstacle_distribution", "unknown"),
            ENUMS["obstacle_distribution"], "obstacle_distribution"
        ),
        "visual_traversal_difficulty": _enum_value(
            terrain_raw.get("visual_traversal_difficulty", "unknown"),
            ENUMS["visual_traversal_difficulty"], "visual_traversal_difficulty"
        ),
        "visual_traversal_reason": traversal_reason.strip(),
        "slope": {
            "assessment": _enum_value(
                slope_raw.get("assessment", "unknown"), ENUMS["slope_assessment"], "slope.assessment"
            ),
            "reason": slope_reason.strip(),
        },
    }

    # Consistency guard: when the dominant surface is smooth, roughness is low,
    # obstacles are isolated and slope is not pronounced, do not let a lone object
    # inflate the whole-scene difficulty. This is a visual grade, not a safety claim.
    if (
        terrain["apparent_roughness"] == "low"
        and terrain["surface_profile"] in {"smooth", "mostly_smooth"}
        and terrain["obstacle_distribution"] == "isolated"
        and terrain["slope"]["assessment"] in {"flat", "nearly_flat", "gentle", "unknown"}
        and terrain["visual_traversal_difficulty"] in {
            "moderate", "moderate_to_difficult", "difficult", "very_difficult"
        }
    ):
        previous_rating = terrain["visual_traversal_difficulty"]
        terrain["visual_traversal_difficulty"] = "easy"
        terrain["visual_traversal_reason"] = (
            "Adjusted to easy for the predominant surface: the visible ground is mostly "
            "smooth with low roughness and isolated obstacles; actual clearance remains unknown."
        )
        LOGGER.warning(
            "Inconsistent terrain grade (%s with smooth/mostly smooth surface, low roughness "
            "and isolated obstacles); normalized to easy. Isolated objects remain recorded.",
            previous_rating,
        )

    observed_objects_raw = payload.get("observed_objects", [])
    if not isinstance(observed_objects_raw, list):
        raise ValueError("`observed_objects` must be a JSON array.")
    observed_objects: list[dict[str, Any]] = []
    for index, observed_object in enumerate(observed_objects_raw):
        if not isinstance(observed_object, dict):
            raise ValueError(f"Observed object #{index} must be a JSON object.")
        label = observed_object.get("label", "")
        description = observed_object.get("description", "")
        confidence = observed_object.get("confidence", 0.0)
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Observed object #{index} needs a non-empty `label`.")
        if not isinstance(description, str):
            raise ValueError(f"Observed object #{index} `description` must be a string.")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise ValueError(f"Observed object #{index} `confidence` must be a number in [0, 1].")
        confidence = float(confidence)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError(f"Observed object #{index} `confidence` must be between 0 and 1.")
        observed_objects.append({
            "label": label.strip(),
            "description": description.strip(),
            "confidence": confidence,
            "bounding_box_normalized": _normalize_box(observed_object.get("bounding_box_normalized")),
        })

    uncertainties = _string_list(payload.get("uncertainties", []), "uncertainties")

    return {
        "photo_id": expected_photo_id,  # Trust the caller's stable ID, not model-generated IDs.
        "terrain": terrain,
        "observed_objects": observed_objects,
        "uncertainties": uncertainties,
    }


class SurveyDatabase:
    """Small SQLite store for image observations; mission scoring belongs above this layer."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS observations (
                source_path TEXT PRIMARY KEY,
                photo_id TEXT NOT NULL,
                image_sha256 TEXT NOT NULL,
                image_modified_at_utc TEXT,
                analyzed_at_utc TEXT NOT NULL,
                latitude REAL,
                longitude REAL,
                model TEXT NOT NULL,
                terrain_json TEXT NOT NULL,
                candidate_objects_json TEXT NOT NULL,
                uncertainties_json TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                CHECK (latitude IS NULL OR (latitude >= -90 AND latitude <= 90)),
                CHECK (longitude IS NULL OR (longitude >= -180 AND longitude <= 180))
            )
            """
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_photo_id ON observations(photo_id)"
        )
        self.connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_observations_analyzed_at ON observations(analyzed_at_utc)"
        )
        self.connection.commit()

    def already_processed(self, source_path: str, image_sha256: str) -> bool:
        row = self.connection.execute(
            "SELECT image_sha256 FROM observations WHERE source_path = ?", (source_path,)
        ).fetchone()
        return row is not None and row["image_sha256"] == image_sha256

    def save(
        self,
        source_path: str,
        image_sha256: str,
        image_modified_at_utc: str,
        model: str,
        observation: dict[str, Any],
    ) -> None:
        terrain_json = json.dumps(observation["terrain"], ensure_ascii=False, separators=(",", ":"))
        # Backward compatibility: the existing SQLite column is still named
        # candidate_objects_json, but now stores the mission-neutral observed_objects list.
        candidates_json = json.dumps(observation["observed_objects"], ensure_ascii=False, separators=(",", ":"))
        uncertainties_json = json.dumps(observation["uncertainties"], ensure_ascii=False, separators=(",", ":"))
        payload_json = json.dumps(observation, ensure_ascii=False, separators=(",", ":"))
        self.connection.execute(
            """
            INSERT INTO observations (
                source_path, photo_id, image_sha256, image_modified_at_utc,
                analyzed_at_utc, model, terrain_json, candidate_objects_json,
                uncertainties_json, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(source_path) DO UPDATE SET
                photo_id = excluded.photo_id,
                image_sha256 = excluded.image_sha256,
                image_modified_at_utc = excluded.image_modified_at_utc,
                analyzed_at_utc = excluded.analyzed_at_utc,
                model = excluded.model,
                terrain_json = excluded.terrain_json,
                candidate_objects_json = excluded.candidate_objects_json,
                uncertainties_json = excluded.uncertainties_json,
                payload_json = excluded.payload_json
            """,
            (
                source_path,
                observation["photo_id"],
                image_sha256,
                image_modified_at_utc,
                utc_now(),
                model,
                terrain_json,
                candidates_json,
                uncertainties_json,
                payload_json,
            ),
        )
        self.connection.commit()

    def update_coordinates(self, source_path: str, latitude: float, longitude: float) -> None:
        """Attach rover/GPS coordinates to an already stored image observation.

        Coordinates should come from rover telemetry or a trusted geotag, not be
        inferred from the image by the vision model.
        """
        if not -90.0 <= latitude <= 90.0:
            raise ValueError("Latitude must be between -90 and 90 degrees.")
        if not -180.0 <= longitude <= 180.0:
            raise ValueError("Longitude must be between -180 and 180 degrees.")
        cursor = self.connection.execute(
            "UPDATE observations SET latitude = ?, longitude = ? WHERE source_path = ?",
            (latitude, longitude, source_path),
        )
        if cursor.rowcount == 0:
            raise KeyError(f"No stored observation found for source path: {source_path}")
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()


def build_client() -> OpenAI:
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "NEBIUS_API_KEY is not set. Add it to your environment or local .env file; "
            "do not hard-code the key in vision_train.py."
        )
    return OpenAI(
        api_key=api_key,
        base_url=os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL).strip(),
        timeout=90.0,
        max_retries=2,
    )


def _extract_message_text(message: Any) -> str:
    """Extract textual content from common OpenAI-compatible response shapes."""
    content = getattr(message, "content", None)
    if isinstance(content, str):
        return content.strip()
    # Some compatible endpoints/SDK versions can represent content as typed parts.
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                part_text = part.get("text")
            else:
                part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                text_parts.append(part_text.strip())
        return "\n".join(text_parts).strip()
    return ""


def analyze_image(client: OpenAI, path: Path, model: str) -> tuple[dict[str, Any], str, str]:
    image_bytes = path.read_bytes()
    digest = sha256_bytes(image_bytes)
    mime_type = image_mime_type(path)
    data_url = f"data:{mime_type};base64,{base64.b64encode(image_bytes).decode('ascii')}"
    photo_id = safe_photo_id(path)

    user_prompt = (
        f"Survey this image for TerraScout. photo_id: {photo_id}.\n"
        "Describe the dominant ground surface and inventory visible objects neutrally using the required JSON schema. "
        "Use the more granular visual difficulty scale and report surface profile and slope separately. "
        "Rate terrain from the predominant surface/distribution, not the largest object. "
        "Do not rank objects by size or future mission relevance; use unknown rather than inventing."
    )
    last_error: Exception | None = None

    # GLM-5.3-Flash may spend most of a small completion budget on hidden reasoning,
    # leaving its JSON answer truncated (finish_reason="length"). Request low reasoning
    # effort and give the structured answer a realistic token budget. The retry keeps
    # reasoning low and raises the budget instead of making truncation more likely.
    for attempt in range(2):
        retry_instruction = ""
        if attempt:
            retry_instruction = (
                "\nRETRY: Your previous output could not be parsed or validated. "
                "Return one compact, complete JSON object only. Include every required "
                "top-level field. Use concise labels, no more than 3 observed_objects, "
                "at most 2 short uncertainties, and short reasons/descriptions. Use empty "
                "arrays when appropriate. Do not use Markdown or extra prose."
            )

        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=4096 if attempt == 0 else 6144,
            response_format={"type": "json_object"},
            # guided_json constrains the visible answer. GLM-5.3-Flash defaults to
            # high/max reasoning on some serving stacks; explicitly request low
            # effort so reasoning does not consume the JSON output budget. This is
            # the model's documented chat_template_kwargs mechanism.
            extra_body={
                "guided_json": GUIDED_JSON_SCHEMA,
                # Include both common GLM serving paths for the same effort level:
                # some compatible servers expose it top-level, others via template kwargs.
                "reasoning_effort": "low",
                "chat_template_kwargs": {"reasoning_effort": "low"},
            },
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT + retry_instruction},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
        )

        if not getattr(response, "choices", None):
            last_error = ValueError("The API response contained no choices.")
        else:
            choice = response.choices[0]
            message = choice.message
            finish_reason = getattr(choice, "finish_reason", None) or "unknown"
            refusal = getattr(message, "refusal", None)
            if isinstance(refusal, str) and refusal.strip():
                raise ValueError(f"The model refused the image request: {refusal.strip()[:300]}")

            content = _extract_message_text(message)
            if not content:
                last_error = ValueError(
                    "The model returned no text content "
                    f"(finish_reason={finish_reason}, "
                    f"content_type={type(getattr(message, 'content', None)).__name__})."
                )
            else:
                try:
                    parsed = parse_json_object(content)
                    normalized = validate_observation(parsed, photo_id)
                except (ValueError, json.JSONDecodeError) as exc:
                    preview = content[:280].replace("\r", " ").replace("\n", " ")
                    usage = getattr(response, "usage", None)
                    completion_tokens = getattr(usage, "completion_tokens", None)
                    last_error = ValueError(
                        f"JSON/schema validation failed (finish_reason={finish_reason}, "
                        f"completion_tokens={completion_tokens}): {exc}. "
                        f"Output preview: {preview!r}"
                    )
                else:
                    modified = datetime.fromtimestamp(
                        path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds")
                    return normalized, digest, modified

        if attempt == 0:
            LOGGER.warning(
                "%s: unusable model output on attempt 1 (%s); retrying with low reasoning and a larger token budget",
                path.name,
                last_error,
            )

    raise ValueError(f"Model output remained unusable after 2 attempts: {last_error}") from last_error


def collect_images(input_path: Path, recursive: bool) -> list[Path]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported image extension: {input_path.suffix}")
        return [input_path]
    iterator = input_path.rglob("*") if recursive else input_path.iterdir()
    return sorted(
        (path for path in iterator if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS),
        key=lambda path: str(path).lower(),
    )


def run(args: argparse.Namespace) -> int:
    input_path = args.input.expanduser().resolve()
    database_path = args.database.expanduser().resolve()
    images = collect_images(input_path, args.recursive)
    if args.limit is not None:
        images = images[: args.limit]
    if not images:
        LOGGER.error("No supported images found at %s", input_path)
        return 2

    model = os.getenv("GLM_MODEL", DEFAULT_MODEL).strip()
    client = build_client()
    database = SurveyDatabase(database_path)
    processed = skipped = failed = 0

    try:
        for index, image_path in enumerate(images, start=1):
            source_path = str(image_path.resolve())
            photo_id = safe_photo_id(image_path)
            try:
                image_bytes = image_path.read_bytes()
                image_digest = sha256_bytes(image_bytes)
                if not args.force and database.already_processed(source_path, image_digest):
                    skipped += 1
                    LOGGER.info("[%d/%d] SKIP %s (unchanged)", index, len(images), image_path.name)
                    continue

                LOGGER.info("[%d/%d] ANALYZE %s", index, len(images), image_path.name)
                observation, digest, modified = analyze_image(client, image_path, model)
                database.save(source_path, digest, modified, model, observation)
                processed += 1
                LOGGER.info(
                    "Saved %s: %d observed object(s); traversal=%s; uncertainty item(s)=%d",
                    observation["photo_id"],
                    len(observation["observed_objects"]),
                    observation["terrain"]["visual_traversal_difficulty"],
                    len(observation["uncertainties"]),
                )
            except (APIConnectionError, APITimeoutError, RateLimitError, APIStatusError) as exc:
                failed += 1
                LOGGER.error("API error for %s: %s", image_path.name, exc)
            except (OSError, ValueError, json.JSONDecodeError, sqlite3.Error) as exc:
                failed += 1
                LOGGER.error("Failed to process %s: %s", image_path.name, exc)
            except Exception:
                failed += 1
                LOGGER.exception("Unexpected error processing %s", image_path.name)
    finally:
        database.close()

    LOGGER.info(
        "Finished. Processed=%d, skipped=%d, failed=%d, database=%s",
        processed, skipped, failed, database_path
    )
    return 1 if failed else 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze survey images with GLM-5.3-Flash and store structured observations in SQLite."
    )
    parser.add_argument(
        "--input", type=Path, required=True,
        help="An image file or a directory containing survey images."
    )
    parser.add_argument(
        "--database", type=Path, default=Path("data/terrascout.db"),
        help="SQLite database path (default: data/terrascout.db)."
    )
    parser.add_argument(
        "--recursive", action="store_true",
        help="Search subdirectories when --input is a directory."
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process at most this many images (useful for a small first test)."
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-analyze images even if their content is already stored in SQLite."
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Enable debug logging."
    )
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        return run(args)
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        LOGGER.error("%s", exc)
        return 2
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
