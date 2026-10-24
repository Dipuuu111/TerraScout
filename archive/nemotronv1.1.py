"""TerraScout mission reasoning with NVIDIA Nemotron via Nebius Token Factory.

Responsibilities:
  * Translate an operator's natural-language mission into a compact mission spec.
  * Score stored image observations deterministically in Python.
  * Ask Nemotron to synthesize a survey/search strategy from the ranked evidence.
  * Optionally watch SQLite for new/changed survey observations and re-plan.

This module does NOT command motors, assert that a region is safe, infer coordinates,
or generate a physically executable route from unlocalized photographs.

Examples (PowerShell):
    python nemotron.py --mission "Find large angular rocks; avoid steep slopes and drop-offs"
    python nemotron.py --mission "Survey the area, prioritizing traversable terrain" --watch
    python nemotron.py --mission "Locate boulders" --database data/terrascout.db --max-regions 8
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # .env loading is optional
    pass

try:
    from openai import OpenAI
except ImportError as exc:  # pragma: no cover - environment dependent
    raise SystemExit(
        "Missing dependency: openai. Install it with `python -m pip install -r requirements-vision.txt`."
    ) from exc

LOGGER = logging.getLogger("terrascout.nemotron")
DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1/"
DEFAULT_MODEL = "nvidia/Nemotron-3_5-Lightning"
DEFAULT_DATABASE = Path("data/terrascout.db")
DEFAULT_OUTPUT_DIR = Path("data/mission_plans")

TRAVERSAL_ORDER = [
    "very_easy", "easy", "easy_to_moderate", "moderate",
    "moderate_to_difficult", "difficult", "very_difficult", "unknown",
]
SLOPE_SEVERITY = {"flat": 0, "nearly_flat": 1, "gentle": 2, "moderate": 3,
                  "steep": 4, "very_steep": 5, "unknown": None}

MISSION_SPEC_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "objective_summary": {"type": "string"},
        "target_keywords": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "preferred_surface_materials": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "preferred_surface_profiles": {"type": "array", "items": {"type": "string", "enum": ["smooth", "mostly_smooth", "gently_uneven", "undulating", "uneven", "broken_rocky", "mixed", "unknown"]}, "maxItems": 8},
        "preferred_traversal_difficulty": {"type": "array", "items": {"type": "string", "enum": TRAVERSAL_ORDER}, "maxItems": 8},
        "avoid_conditions": {"type": "array", "items": {"type": "object", "properties": {
            "condition": {"type": "string", "enum": ["clear_drop_off", "possible_drop_off", "steep_slope", "moderate_slope", "high_roughness", "widespread_obstacles", "continuous_obstacles", "loose_surface", "unknown_terrain"]},
            "severity": {"type": "string", "enum": ["hard", "soft"]}
        }, "required": ["condition", "severity"], "additionalProperties": False}, "maxItems": 9},
        "priorities": {"type": "array", "items": {"type": "string", "enum": ["target_match", "surface_preference", "traversability", "coverage", "risk_avoidance"]}, "maxItems": 5},
        "operator_constraints": {"type": "array", "items": {"type": "string"}, "maxItems": 10}
    },
    "required": ["objective_summary", "target_keywords", "preferred_surface_materials", "preferred_surface_profiles", "preferred_traversal_difficulty", "avoid_conditions", "priorities", "operator_constraints"],
    "additionalProperties": False,
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "mission_title": {"type": "string"},
        "mission_summary": {"type": "string"},
        "status": {"type": "string", "enum": ["ready_for_survey_review", "insufficient_data", "needs_operator_input"]},
        "selected_regions": {"type": "array", "items": {"type": "object", "properties": {
            "photo_id": {"type": "string"}, "purpose": {"type": "string"}, "reason": {"type": "string"}
        }, "required": ["photo_id", "purpose", "reason"], "additionalProperties": False}, "maxItems": 10},
        "search_strategy": {"type": "array", "items": {"type": "object", "properties": {
            "step": {"type": "integer"}, "action": {"type": "string"}, "based_on_photo_ids": {"type": "array", "items": {"type": "string"}}, "why": {"type": "string"}, "completion_condition": {"type": "string"}
        }, "required": ["step", "action", "based_on_photo_ids", "why", "completion_condition"], "additionalProperties": False}, "maxItems": 8},
        "replan_triggers": {"type": "array", "items": {"type": "string"}, "maxItems": 8},
        "uncertainties": {"type": "array", "items": {"type": "string"}, "maxItems": 12},
        "operator_questions": {"type": "array", "items": {"type": "string"}, "maxItems": 6},
        "navigation_readiness_reason": {"type": "string"}
    },
    "required": ["mission_title", "mission_summary", "status", "selected_regions", "search_strategy", "replan_triggers", "uncertainties", "operator_questions", "navigation_readiness_reason"],
    "additionalProperties": False,
}

MISSION_SYSTEM_PROMPT = """You are TerraScout's mission-intent compiler. Convert the operator's stated mission into the supplied JSON schema.
Return JSON only. Extract object targets as short canonical keywords rather than long phrases (for example, ["large", "angular", "rock"] instead of ["large angular rocks"]); use singular object nouns where natural. Do not invent a goal. Use controlled surface-profile and traversal enums exactly as provided by the schema. Add avoid_conditions only when the operator explicitly asks to avoid a hazard or condition, or the task clearly implies safety-first traversal. Use severity=hard only for an explicit prohibition; otherwise use soft. Do not treat a detected object as a mission target unless the operator asks for it. Do not infer coordinates, rover capabilities, distances, or safety guarantees. Preserve important operator constraints as short strings."""

PLANNER_SYSTEM_PROMPT = """You are TerraScout's mission-reasoning and survey-planning model. You receive an operator mission, a structured mission specification, observations analyzed by a separate vision model, and deterministic Python ranking results. Synthesize a concise evidence-grounded strategy. Python owns numeric scoring; do not invent or alter scores. Reference only photo_id values supplied in the ranked evidence. Treat a photo as a survey observation, not proof that its scene is reachable or spatially adjacent to another photo. Never invent coordinates, distances, adjacency, physical dimensions, or rover capabilities. Terrain ratings and hazard cues are visual inferences, not verified navigation safety. If positions are absent, propose survey/reconnaissance steps and say a physical route cannot yet be computed. Never output motor commands. Identify conflicting evidence and missing information explicitly. Return one JSON object matching the schema, no Markdown or surrounding prose."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    start = cleaned.find("{")
    if start < 0:
        raise ValueError("Nemotron response did not contain a JSON object.")
    value, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    if not isinstance(value, dict):
        raise ValueError("Nemotron JSON root must be an object.")
    return value


def compact_text(value: Any, *, max_len: int = 400) -> str:
    if not isinstance(value, str):
        return ""
    return re.sub(r"\s+", " ", value).strip()[:max_len]


def _string_list(value: Any, field: str, limit: int = 30) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array.")
    result: list[str] = []
    for item in value[:limit]:
        if isinstance(item, str) and item.strip():
            result.append(item.strip()[:160])
    return result


def normalize_mission_spec(raw: dict[str, Any]) -> dict[str, Any]:
    allowed_profiles = {"smooth", "mostly_smooth", "gently_uneven", "undulating", "uneven", "broken_rocky", "mixed", "unknown"}
    allowed_traversal = set(TRAVERSAL_ORDER)
    allowed_avoid = {"clear_drop_off", "possible_drop_off", "steep_slope", "moderate_slope", "high_roughness", "widespread_obstacles", "continuous_obstacles", "loose_surface", "unknown_terrain"}
    allowed_priorities = {"target_match", "surface_preference", "traversability", "coverage", "risk_avoidance"}

    avoids: list[dict[str, str]] = []
    raw_avoids = raw.get("avoid_conditions", [])
    if not isinstance(raw_avoids, list):
        raise ValueError("avoid_conditions must be an array.")
    for item in raw_avoids:
        if not isinstance(item, dict):
            continue
        condition, severity = item.get("condition"), item.get("severity")
        if condition in allowed_avoid and severity in {"hard", "soft"}:
            if not any(existing["condition"] == condition for existing in avoids):
                avoids.append({"condition": condition, "severity": severity})

    def enum_list(key: str, allowed: set[str]) -> list[str]:
        values = _string_list(raw.get(key, []), key, 16)
        return list(dict.fromkeys(v for v in values if v in allowed))

    priorities = enum_list("priorities", allowed_priorities)
    if not priorities:
        priorities = ["target_match", "risk_avoidance", "surface_preference", "traversability", "coverage"]

    summary = compact_text(raw.get("objective_summary"), max_len=800)
    if not summary:
        raise ValueError("Nemotron returned an empty objective_summary.")
    return {
        "objective_summary": summary,
        "target_keywords": list(dict.fromkeys(_string_list(raw.get("target_keywords", []), "target_keywords", 12))),
        "preferred_surface_materials": list(dict.fromkeys(_string_list(raw.get("preferred_surface_materials", []), "preferred_surface_materials", 8))),
        "preferred_surface_profiles": enum_list("preferred_surface_profiles", allowed_profiles),
        "preferred_traversal_difficulty": enum_list("preferred_traversal_difficulty", allowed_traversal),
        "avoid_conditions": avoids,
        "priorities": priorities,
        "operator_constraints": list(dict.fromkeys(_string_list(raw.get("operator_constraints", []), "operator_constraints", 10))),
    }


def open_client() -> OpenAI:
    api_key = os.getenv("NEBIUS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("NEBIUS_API_KEY is not set. Put it in your local .env file; do not hard-code API keys.")
    base_url = os.getenv("NEBIUS_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)


def call_json(client: OpenAI, model: str, system_prompt: str, user_prompt: str, *, stage: str, max_tokens: int, validator: Callable[[dict[str, Any]], Any] | None = None) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(2):
        retry_note = ""
        if attempt:
            retry_note = " Previous response was invalid; return a shorter, complete JSON object and no prose."
        response = client.chat.completions.create(
            model=model,
            temperature=0.1,
            max_tokens=max_tokens if not attempt else max_tokens + 1500,
            response_format={"type": "json_object"},
            # Nemotron 3.5 Lightning enables thinking by default on compatible
            # serving stacks. Thinking tokens share max_tokens with the visible
            # answer and can consume the entire budget before JSON is produced.
            # Disable thinking for both structured-output calls; the task needs
            # a compact machine-readable result, not a reasoning trace.
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            messages=[
                {"role": "system", "content": system_prompt + retry_note},
                {"role": "user", "content": user_prompt},
            ],
        )
        choices = getattr(response, "choices", None) or []
        if not choices:
            last_error = ValueError(f"{stage}: API returned no choices.")
            continue
        choice = choices[0]
        content = getattr(choice.message, "content", None)
        finish_reason = getattr(choice, "finish_reason", None)
        if not isinstance(content, str) or not content.strip():
            last_error = ValueError(f"{stage}: model returned empty content (finish_reason={finish_reason}).")
            continue
        usage = getattr(response, "usage", None)
        completion_tokens = getattr(usage, "completion_tokens", None) if usage else None
        # A length-truncated answer is not accepted, even if its prefix happens
        # to look like a parseable object: do not plan from partial model output.
        if finish_reason == "length":
            preview = content[:220].replace("\n", " ")
            last_error = ValueError(
                f"{stage}: response was truncated (finish_reason=length, "
                f"completion_tokens={completion_tokens}); preview={preview!r}"
            )
            LOGGER.warning("%s attempt %d failed: %s", stage, attempt + 1, last_error)
            continue
        try:
            parsed = parse_json_object(content)
            # Validate required semantic fields inside the retry loop too. JSON
            # mode can still produce an object with empty required strings.
            if validator is not None:
                validator(parsed)
            return parsed
        except (ValueError, json.JSONDecodeError) as exc:
            preview = content[:220].replace("\n", " ")
            last_error = ValueError(
                f"{stage}: invalid JSON (finish_reason={finish_reason}, "
                f"completion_tokens={completion_tokens}): {exc}; preview={preview!r}"
            )
            LOGGER.warning("%s attempt %d failed: %s", stage, attempt + 1, last_error)
    raise RuntimeError(f"{stage} failed after 2 attempts: {last_error}") from last_error


def parse_mission(client: OpenAI, model: str, mission: str) -> dict[str, Any]:
    user_prompt = (
        "Compile this operator mission into the required schema. Never infer physical capability or location.\n\n"
        f"OPERATOR MISSION:\n{mission}\n\nSCHEMA:\n{json.dumps(MISSION_SPEC_SCHEMA, separators=(',', ':'))}"
    )
    return normalize_mission_spec(call_json(client, model, MISSION_SYSTEM_PROMPT, user_prompt, stage="mission-intent parsing", max_tokens=2200, validator=normalize_mission_spec))


def load_observations(database_path: Path) -> list[dict[str, Any]]:
    if not database_path.is_file():
        raise FileNotFoundError(f"Survey database not found: {database_path}. Start main.py and ingest at least one image first.")
    connection = sqlite3.connect(str(database_path), timeout=10.0)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='observations'").fetchone()
        if table is None:
            raise RuntimeError(f"Database has no observations table: {database_path}")
        rows = connection.execute(
            """SELECT source_path, photo_id, image_sha256, image_modified_at_utc,
                      analyzed_at_utc, latitude, longitude, model, terrain_json,
                      candidate_objects_json, uncertainties_json, payload_json
               FROM observations ORDER BY analyzed_at_utc ASC"""
        ).fetchall()
    finally:
        connection.close()

    observations: list[dict[str, Any]] = []
    for row in rows:
        try:
            payload = json.loads(row["payload_json"] or "{}")
            terrain = json.loads(row["terrain_json"] or "{}")
            objects = json.loads(row["candidate_objects_json"] or "[]")
            uncertainties = json.loads(row["uncertainties_json"] or "[]")
        except json.JSONDecodeError:
            LOGGER.warning("Skipping malformed database record: %s", row["photo_id"])
            continue
        if not isinstance(terrain, dict):
            terrain = {}
        if not isinstance(objects, list):
            objects = []
        if not isinstance(uncertainties, list):
            uncertainties = []
        observations.append({
            "photo_id": str(row["photo_id"]),
            "source_path": str(row["source_path"]),
            "image_sha256": str(row["image_sha256"]),
            "analyzed_at_utc": row["analyzed_at_utc"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "model": row["model"],
            "terrain": terrain,
            "observed_objects": objects,
            "uncertainties": uncertainties,
            "payload": payload if isinstance(payload, dict) else {},
        })
    return observations


def norm(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def has_phrase(text: str, term: str) -> bool:
    """Match a normalized phrase, tolerating simple singular/plural noun variants."""
    text_norm = f" {norm(text)} "
    term_norm = norm(term)
    if not term_norm:
        return False
    variants = {term_norm}
    words = term_norm.split()
    if words:
        last = words[-1]
        stems: set[str] = set()
        if last.endswith("ies") and len(last) > 3:
            stems.add(last[:-3] + "y")
        if last.endswith("s") and not last.endswith(("ss", "us", "is")) and len(last) > 2:
            stems.add(last[:-1])
        else:
            stems.update({last + "s", last + "es"})
        for stem in stems:
            variants.add(" ".join(words[:-1] + [stem]))
    return any(f" {variant} " in text_norm for variant in variants)


def build_searchable_text(observation: dict[str, Any]) -> tuple[str, str]:
    terrain = observation.get("terrain", {})
    objects = observation.get("observed_objects", [])
    object_text_parts: list[str] = []
    for obj in objects:
        if isinstance(obj, dict):
            object_text_parts.extend([str(obj.get("label", "")), str(obj.get("description", ""))])
    object_text = " ".join(object_text_parts)
    terrain_fields: list[str] = []
    for key in ("surface_materials", "rock_shapes", "dominant_colors", "secondary_colors"):
        value = terrain.get(key, [])
        if isinstance(value, list):
            terrain_fields.extend(str(item) for item in value)
    for key in ("surface_profile", "relative_rock_abundance", "apparent_roughness", "obstacle_distribution", "visual_traversal_difficulty"):
        terrain_fields.append(str(terrain.get(key, "")))
    slope = terrain.get("slope", {})
    dropoff = terrain.get("drop_off", {})
    if isinstance(slope, dict):
        terrain_fields.extend([str(slope.get("assessment", "")), str(slope.get("direction", "")), str(slope.get("reason", ""))])
    if isinstance(dropoff, dict):
        terrain_fields.extend([str(dropoff.get("assessment", "")), str(dropoff.get("reason", ""))])
    return norm(" ".join(terrain_fields)), norm(object_text)


def evaluate_avoid_conditions(observation: dict[str, Any], mission_spec: dict[str, Any]) -> tuple[float, list[str], list[str]]:
    terrain = observation.get("terrain", {})
    slope = terrain.get("slope", {}) if isinstance(terrain.get("slope"), dict) else {}
    drop = terrain.get("drop_off", {}) if isinstance(terrain.get("drop_off"), dict) else {}
    penalties: list[str] = []
    hard_violations: list[str] = []
    penalty = 0.0
    avoid_conditions = {item["condition"]: item["severity"] for item in mission_spec.get("avoid_conditions", [])}
    priorities = mission_spec.get("priorities", [])
    risk_rank = priorities.index("risk_avoidance") if "risk_avoidance" in priorities else len(priorities)
    risk_multiplier = 1.25 if risk_rank <= 1 else 1.0 if risk_rank == 2 else 0.8

    def apply(condition: str, amount: float, evidence: str) -> None:
        nonlocal penalty
        severity = avoid_conditions.get(condition)
        if severity:
            penalty += amount * (1.5 if severity == "hard" else 1.0) * risk_multiplier
            message = f"{condition}: {evidence} ({severity} avoid condition)"
            penalties.append(message)
            if severity == "hard":
                hard_violations.append(message)

    drop_assessment = drop.get("assessment", "unknown")
    if drop_assessment == "clear":
        apply("clear_drop_off", 48, "drop-off marked clear in visual assessment")
    elif drop_assessment == "possible":
        apply("possible_drop_off", 25, "possible drop-off reported visually")
    elif drop_assessment == "unknown":
        apply("unknown_terrain", 7, "drop-off status unknown")

    slope_assessment = slope.get("assessment", "unknown")
    if slope_assessment in {"steep", "very_steep"}:
        apply("steep_slope", 30 if slope_assessment == "steep" else 40, f"slope assessment={slope_assessment}")
    elif slope_assessment == "moderate":
        apply("moderate_slope", 14, "slope assessment=moderate")
    elif slope_assessment == "unknown":
        apply("unknown_terrain", 6, "slope assessment unknown")

    if terrain.get("apparent_roughness") == "high":
        apply("high_roughness", 18, "apparent roughness=high")
    elif terrain.get("apparent_roughness") == "unknown":
        apply("unknown_terrain", 4, "apparent roughness unknown")

    distribution = terrain.get("obstacle_distribution")
    if distribution == "widespread":
        apply("widespread_obstacles", 15, "obstacles are widespread")
    elif distribution == "continuous":
        apply("continuous_obstacles", 25, "obstacles are continuous")

    materials = terrain.get("surface_materials", [])
    material_text = norm(" ".join(str(x) for x in materials))
    if any(term in material_text for term in ("loose", "sand", "gravel", "scree", "dust")):
        apply("loose_surface", 12, "surface materials may be loose")

    return penalty, penalties, hard_violations


def score_observation(observation: dict[str, Any], mission_spec: dict[str, Any]) -> dict[str, Any]:
    """Transparent deterministic matching/risk score; not a safety probability."""
    terrain = observation.get("terrain", {})
    all_text, object_text = build_searchable_text(observation)
    target_keywords = mission_spec.get("target_keywords", [])
    target_hits = [kw for kw in target_keywords if has_phrase(object_text, kw) or has_phrase(all_text, kw)]
    priorities = mission_spec.get("priorities", [])
    def priority_factor(name: str) -> float:
        rank = priorities.index(name) if name in priorities else len(priorities)
        return (1.25, 1.10, 0.95, 0.80, 0.65)[min(rank, 4)]

    target_weight = 45.0 * priority_factor("target_match") if target_keywords else 0.0
    target_component = target_weight * len(target_hits) / len(target_keywords) if target_keywords else 0.0

    pref_materials = mission_spec.get("preferred_surface_materials", [])
    material_hits = [term for term in pref_materials if has_phrase(all_text, term)]
    material_weight = 15.0 * priority_factor("surface_preference") if pref_materials else 0.0
    material_component = material_weight * len(material_hits) / len(pref_materials) if pref_materials else 0.0

    pref_profiles = mission_spec.get("preferred_surface_profiles", [])
    profile = str(terrain.get("surface_profile", "unknown"))
    profile_hits = [v for v in pref_profiles if v == profile]
    profile_weight = 10.0 * priority_factor("surface_preference") if pref_profiles else 0.0
    profile_component = profile_weight * len(profile_hits) / len(pref_profiles) if pref_profiles else 0.0

    pref_traversal = mission_spec.get("preferred_traversal_difficulty", [])
    traversal = str(terrain.get("visual_traversal_difficulty", "unknown"))
    traversal_hits = [v for v in pref_traversal if v == traversal]
    traversal_weight = 10.0 * priority_factor("traversability") if pref_traversal else 0.0
    traversal_component = traversal_weight * len(traversal_hits) / len(pref_traversal) if pref_traversal else 0.0

    # Confidence/data completeness help compare otherwise similar evidence, but are
    # intentionally low-weight and are not interpreted as a probability of safety.
    objects = observation.get("observed_objects", [])
    confidences = [float(o.get("confidence", 0.0)) for o in objects if isinstance(o, dict) and isinstance(o.get("confidence"), (int, float))]
    confidence_component = (sum(confidences) / len(confidences) * 8.0) if confidences else 0.0
    data_component = 4.0 if terrain and traversal != "unknown" else 0.0
    coord_component = 3.0 if observation.get("latitude") is not None and observation.get("longitude") is not None else 0.0

    components = [
        (target_component, target_weight), (material_component, material_weight),
        (profile_component, profile_weight), (traversal_component, traversal_weight),
        (confidence_component, 8.0), (data_component, 4.0), (coord_component, 3.0),
    ]
    total_possible = sum(weight for _, weight in components)
    positive = sum(value for value, _ in components)
    raw_score = 100.0 * positive / total_possible if total_possible else 0.0
    penalty, penalty_reasons, hard_violations = evaluate_avoid_conditions(observation, mission_spec)
    score = max(0.0, min(100.0, raw_score - penalty))
    return {
        "photo_id": observation["photo_id"],
        "source_path": observation["source_path"],
        "analyzed_at_utc": observation.get("analyzed_at_utc"),
        "coordinates": {"latitude": observation.get("latitude"), "longitude": observation.get("longitude")},
        "score": round(score, 2),
        "score_components": {
            "target_keyword_hits": target_hits,
            "preferred_material_hits": material_hits,
            "preferred_profile_hits": profile_hits,
            "preferred_traversal_hits": traversal_hits,
            "mean_object_confidence": round(sum(confidences) / len(confidences), 3) if confidences else None,
            "raw_match_score": round(raw_score, 2),
            "avoidance_penalty": round(penalty, 2),
        },
        "avoidance_notes": penalty_reasons,
        "hard_constraint_violations": hard_violations,
        "eligible_for_priority": not bool(hard_violations),
        "evidence": {
            "surface_materials": terrain.get("surface_materials", []),
            "surface_profile": terrain.get("surface_profile", "unknown"),
            "relative_rock_abundance": terrain.get("relative_rock_abundance", "unknown"),
            "apparent_roughness": terrain.get("apparent_roughness", "unknown"),
            "obstacle_distribution": terrain.get("obstacle_distribution", "unknown"),
            "visual_traversal_difficulty": traversal,
            "visual_traversal_reason": terrain.get("visual_traversal_reason", ""),
            "slope": terrain.get("slope", {}),
            "drop_off": terrain.get("drop_off", {}),
            "observed_objects": objects[:8],
            "uncertainties": observation.get("uncertainties", [])[:8],
        },
    }


def rank_observations(observations: list[dict[str, Any]], mission_spec: dict[str, Any], max_regions: int) -> list[dict[str, Any]]:
    ranked = [score_observation(o, mission_spec) for o in observations]
    # Hard-constraint violations are never prioritized as preferred destinations.
    ranked.sort(key=lambda r: (r["eligible_for_priority"], r["score"], r["analyzed_at_utc"] or ""), reverse=True)
    return ranked[:max_regions]


def _safe_plan(plan: dict[str, Any], ranked: list[dict[str, Any]], observations: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize output, discard unknown photo IDs, and reject hard-conflict destinations."""
    valid_photo_ids = {item["photo_id"] for item in ranked}
    eligible_photo_ids = {item["photo_id"] for item in ranked if item.get("eligible_for_priority", True)}
    allowed_status = {"ready_for_survey_review", "insufficient_data", "needs_operator_input"}
    status = plan.get("status") if plan.get("status") in allowed_status else "insufficient_data"
    selected: list[dict[str, str]] = []
    for item in plan.get("selected_regions", []) if isinstance(plan.get("selected_regions"), list) else []:
        if not isinstance(item, dict) or item.get("photo_id") not in valid_photo_ids:
            continue
        # A model-generated plan may not promote an observation with a deterministic
        # hard-constraint conflict to a selected destination. It may still reference
        # that photo in a step as evidence of a hazard to avoid.
        if item.get("photo_id") not in eligible_photo_ids:
            continue
        selected.append({
            "photo_id": item["photo_id"],
            "purpose": compact_text(item.get("purpose"), max_len=240),
            "reason": compact_text(item.get("reason"), max_len=500),
        })
    steps: list[dict[str, Any]] = []
    for i, item in enumerate(plan.get("search_strategy", []) if isinstance(plan.get("search_strategy"), list) else [], start=1):
        if not isinstance(item, dict):
            continue
        refs = [p for p in item.get("based_on_photo_ids", []) if isinstance(p, str) and p in valid_photo_ids] if isinstance(item.get("based_on_photo_ids", []), list) else []
        steps.append({
            "step": i,
            "action": compact_text(item.get("action"), max_len=300),
            "based_on_photo_ids": refs,
            "why": compact_text(item.get("why"), max_len=500),
            "completion_condition": compact_text(item.get("completion_condition"), max_len=300),
        })
    has_full_coordinates = bool(observations) and all(o.get("latitude") is not None and o.get("longitude") is not None for o in observations)
    nav_reason = compact_text(plan.get("navigation_readiness_reason"), max_len=700)
    if not has_full_coordinates:
        nav_reason = (nav_reason + " " if nav_reason else "") + "At least one stored observation lacks trusted latitude/longitude."
    nav_reason = (nav_reason + " " if nav_reason else "") + "This module does not compute a physical route or issue movement commands; coordinates alone are insufficient to establish a safe path."
    if ranked and not eligible_photo_ids:
        status = "needs_operator_input"
        uncertainty_note = "Every ranked observation conflicts with at least one hard mission avoid-condition; no observation was selected as a preferred destination."
    else:
        uncertainty_note = ""
    uncertainties = _string_list(plan.get("uncertainties", []) if isinstance(plan.get("uncertainties", []), list) else [], "uncertainties", 12)
    if uncertainty_note and uncertainty_note not in uncertainties:
        uncertainties.append(uncertainty_note)
    replan_triggers = _string_list(plan.get("replan_triggers", []) if isinstance(plan.get("replan_triggers", []), list) else [], "replan_triggers", 8)
    operator_questions = _string_list(plan.get("operator_questions", []) if isinstance(plan.get("operator_questions", []), list) else [], "operator_questions", 6)
    return {
        "mission_title": compact_text(plan.get("mission_title"), max_len=180) or "TerraScout mission plan",
        "mission_summary": compact_text(plan.get("mission_summary"), max_len=1000),
        "status": status,
        "selected_regions": selected[:10],
        "search_strategy": steps[:8],
        "replan_triggers": replan_triggers,
        "uncertainties": uncertainties,
        "operator_questions": operator_questions,
        "navigation_readiness_reason": nav_reason,
        "georeferenced_observations_available": has_full_coordinates,
        "physical_route_computable": False,
        "movement_commands_emitted": False,
    }


def synthesize_plan(client: OpenAI, model: str, mission: str, mission_spec: dict[str, Any], observations: list[dict[str, Any]], ranked: list[dict[str, Any]], previous_plan: dict[str, Any] | None = None) -> dict[str, Any]:
    if not observations:
        return {
            "mission_title": "TerraScout mission plan", "mission_summary": "No survey observations are available yet.",
            "status": "insufficient_data", "selected_regions": [], "search_strategy": [],
            "replan_triggers": ["Re-plan when a new image has been analysed and stored."],
            "uncertainties": ["The survey database has no analyzed observations."],
            "operator_questions": [], "navigation_readiness_reason": "No survey data is available.",
            "georeferenced_observations_available": False, "physical_route_computable": False, "movement_commands_emitted": False,
        }
    # Do not disclose local source paths to the model; photo IDs and analysis evidence suffice.
    ranked_for_model = [{
        "photo_id": item["photo_id"],
        "score": item["score"],
        "score_components": item["score_components"],
        "avoidance_notes": item["avoidance_notes"],
        "hard_constraint_violations": item["hard_constraint_violations"],
        "eligible_for_priority": item["eligible_for_priority"],
        "evidence": item["evidence"],
    } for item in ranked]
    evidence_prompt = {
        "operator_mission": mission,
        "mission_spec": mission_spec,
        "deterministic_python_ranking": ranked_for_model,
        "previous_plan_for_replanning": previous_plan,
        "instructions": [
            "Explain why the highest-ranked regions fit the objective; Python computed ranking scores.",
            "Do not alter numeric scores or invent observations.",
            "Use only photo_id values present in deterministic_python_ranking.",
            "If hard constraint violations exist, do not recommend those observations as preferred destinations; mention the visual hazard and uncertainty.",
            "If coordinates are missing, plan further survey/inspection of records, not a physical route.",
            "Propose concrete next observations and explain what new evidence would change the plan.",
            "Do not emit motor commands or declare a region safe.",
        ],
        "output_schema": PLAN_SCHEMA,
    }
    raw = call_json(client, model, PLANNER_SYSTEM_PROMPT, json.dumps(evidence_prompt, ensure_ascii=False, separators=(",", ":")), stage="mission planning", max_tokens=3200)
    return _safe_plan(raw, ranked, observations)


def snapshot_signature(database_path: Path) -> str:
    if not database_path.is_file():
        return "missing"
    connection = sqlite3.connect(str(database_path), timeout=10.0)
    try:
        rows = connection.execute("SELECT source_path, image_sha256, analyzed_at_utc FROM observations ORDER BY source_path").fetchall()
    except sqlite3.Error:
        return "unavailable"
    finally:
        connection.close()
    content = "\n".join("|".join(str(v or "") for v in row) for row in rows)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def save_plan(output_dir: Path, mission: str, mission_spec: dict[str, Any], ranked: list[dict[str, Any]], plan: dict[str, Any], revision: int) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", mission.lower()).strip("-")[:48] or "mission"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    document = {
        "schema_version": 1,
        "revision": revision,
        "created_at_utc": utc_now(),
        "operator_mission": mission,
        "mission_spec": mission_spec,
        "deterministic_ranked_regions": ranked,
        "plan": plan,
    }
    destination = output_dir / f"{slug}_r{revision:03d}_{stamp}.json"
    destination.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    latest = output_dir / f"{slug}_latest.json"
    latest.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
    LOGGER.info("Saved mission plan: %s", destination.resolve())
    return destination


def print_report(mission_spec: dict[str, Any], ranked: list[dict[str, Any]], plan: dict[str, Any], revision: int) -> None:
    print("\n" + "=" * 76)
    print(f"TERRASCOUT MISSION PLAN — REVISION {revision}")
    print("=" * 76)
    print(f"Mission: {mission_spec['objective_summary']}")
    print(f"Status:  {plan.get('status', 'unknown')}")
    print("\nPython deterministic ranking (scores are prioritization signals, not safety probabilities):")
    if not ranked:
        print("  No survey observations available.")
    for index, region in enumerate(ranked, 1):
        mark = " [HARD CONSTRAINT CONFLICT]" if region["hard_constraint_violations"] else ""
        coords = region["coordinates"]
        loc = f" ({coords['latitude']}, {coords['longitude']})" if coords["latitude"] is not None and coords["longitude"] is not None else " (unlocalized)"
        print(f"  {index:>2}. {region['photo_id']}: score={region['score']:.1f}{mark}{loc}")
        for note in region["avoidance_notes"]:
            print(f"      - {note}")
    print("\nNemotron strategy:")
    print(plan.get("mission_summary", "(no summary)"))
    for region in plan.get("selected_regions", []):
        print(f"  • {region['photo_id']}: {region['purpose']} — {region['reason']}")
    for step in plan.get("search_strategy", []):
        refs = ", ".join(step["based_on_photo_ids"]) or "no specific image reference"
        print(f"  {step['step']}. {step['action']} [{refs}]")
        print(f"     Why: {step['why']}")
        if step["completion_condition"]:
            print(f"     Complete when: {step['completion_condition']}")
    if plan.get("uncertainties"):
        print("\nUncertainties:")
        for item in plan["uncertainties"]:
            print(f"  - {item}")
    if plan.get("operator_questions"):
        print("\nOperator input requested:")
        for item in plan["operator_questions"]:
            print(f"  - {item}")
    print(f"\nPhysical route computable: {plan.get('physical_route_computable', False)}")
    print(f"Navigation note: {plan.get('navigation_readiness_reason', '')}")
    print("Movement commands emitted: False")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TerraScout mission reasoning and replanning using NVIDIA Nemotron.")
    parser.add_argument("--mission", required=True, help="Operator's natural-language mission objective.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE, help="Survey SQLite DB (default: data/terrascout.db).")
    parser.add_argument("--model", default=os.getenv("NEMOTRON_MODEL", DEFAULT_MODEL), help=f"Nemotron model ID (default: {DEFAULT_MODEL}).")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Where mission plan JSON snapshots are stored.")
    parser.add_argument("--max-regions", type=int, default=10, help="Number of top-ranked observations to send to Nemotron (1-30).")
    parser.add_argument("--watch", action="store_true", help="Watch SQLite for newly analysed/changed observations and re-plan the same mission.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="DB polling interval for --watch (default: 5s).")
    parser.add_argument("--debounce-seconds", type=float, default=4.0, help="Wait for DB changes to settle before replanning (default: 4s).")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logs.")
    return parser


def run_once(client: OpenAI, model: str, mission: str, mission_spec: dict[str, Any], database_path: Path, output_dir: Path, max_regions: int, previous_plan: dict[str, Any] | None, revision: int) -> dict[str, Any]:
    observations = load_observations(database_path) if database_path.is_file() else []
    ranked = rank_observations(observations, mission_spec, max_regions)
    plan = synthesize_plan(client, model, mission, mission_spec, observations, ranked, previous_plan)
    save_plan(output_dir, mission, mission_spec, ranked, plan, revision)
    print_report(mission_spec, ranked, plan, revision)
    return plan


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.mission.strip():
        parser.error("--mission cannot be empty")
    if not 1 <= args.max_regions <= 30:
        parser.error("--max-regions must be between 1 and 30")
    if args.poll_seconds <= 0 or args.debounce_seconds < 0:
        parser.error("--poll-seconds must be positive and --debounce-seconds cannot be negative")
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    database_path = args.database.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    model = args.model.strip()

    try:
        client = open_client()
        LOGGER.info("Compiling mission intent with %s", model)
        mission_spec = parse_mission(client, model, args.mission.strip())
        LOGGER.info("Mission objective parsed: %s", mission_spec["objective_summary"])
        previous_plan: dict[str, Any] | None = None
        revision = 1
        current_sig = snapshot_signature(database_path)
        observations = load_observations(database_path) if database_path.is_file() else []
        if observations:
            previous_plan = run_once(client, model, args.mission.strip(), mission_spec, database_path, output_dir, args.max_regions, None, revision)
        elif not args.watch:
            raise RuntimeError(f"No survey observations found in {database_path}. Run main.py and wait for at least one image analysis to finish.")
        else:
            LOGGER.info("No observations stored yet; waiting for main.py to analyse the first image.")

        if args.watch:
            LOGGER.info("Watching the survey database for new/changed observations. Press Ctrl+C to stop.")
            while True:
                time.sleep(args.poll_seconds)
                new_sig = snapshot_signature(database_path)
                if new_sig == current_sig or new_sig in {"missing", "unavailable"}:
                    continue
                # Debounce a burst of DB writes from the image watcher. Continue watching
                # until the signature remains unchanged for the configured period.
                settle_deadline = time.monotonic() + args.debounce_seconds
                settled_sig = new_sig
                while time.monotonic() < settle_deadline:
                    time.sleep(min(args.poll_seconds, max(0.1, settle_deadline - time.monotonic())))
                    candidate_sig = snapshot_signature(database_path)
                    if candidate_sig != settled_sig:
                        settled_sig = candidate_sig
                        settle_deadline = time.monotonic() + args.debounce_seconds
                if settled_sig == current_sig:
                    continue
                current_sig = settled_sig
                revision += 1
                LOGGER.info("Survey DB changed; replanning mission (revision %d).", revision)
                previous_plan = run_once(client, model, args.mission.strip(), mission_spec, database_path, output_dir, args.max_regions, previous_plan, revision)
    except KeyboardInterrupt:
        LOGGER.info("Stopped by operator.")
        return 0
    except Exception as exc:
        LOGGER.error("Mission reasoning failed: %s", exc)
        if args.verbose:
            LOGGER.exception("Debug traceback")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
