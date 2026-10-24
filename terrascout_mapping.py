"""TerraScout's sensor-aware grid mapping foundation.

This module intentionally keeps two map layers separate:

1. Per-image normalized grids: show where GLM detections sit inside each image.
   These grids are NOT metric and different photographs are NOT spatially aligned.
2. A sparse metric occupancy/coverage grid: populated only from measurements with a
   valid pose and a known measurement convention (e.g. LiDAR range scans, depth or
   point-cloud data). GPS-tagged photos can be placed as survey-location markers.

Standard sensor-frame JSON format (one JSON file per frame):
{
  "schema_version": 1,
  "frame_id": "front_lidar_000123",
  "photo_id": "image_001",                 // optional association
  "timestamp_utc": "2026-10-10T12:00:00Z",
  "sensor_type": "lidar_2d",
  "pose": {                                  // sensor origin in shared map frame
    "x_m": 0.0, "y_m": 0.0, "z_m": 0.25,
    "yaw_deg": 0.0, "pitch_deg": 0.0, "roll_deg": 0.0,
    "quality": "estimated", "source": "wheel_odometry+imu"
  },
  "range_scan": {
    "angle_min_deg": -90.0, "angle_increment_deg": 1.0,
    "range_min_m": 0.10, "range_max_m": 12.0,
    "ranges_m": [2.0, 2.1, null, 12.0]
  }
}

For 3-D LiDAR, depth cameras or stereo, adapters can instead provide `points` in
sensor-pose-aligned coordinates: x forward, y left, z up, all in metres. `pose`
then places that sensor coordinate origin/orientation in the shared map frame.

Never infer metres from an ordinary monocular RGB image or GLM bounding boxes alone.
A calibrated monocular camera may estimate flat-ground coverage if camera intrinsics,
height, pitch and pose are supplied; this is explicitly labelled as an assumption and
adds coverage evidence only, never free-space/obstacle-free evidence.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger("terrascout.mapping")

DEFAULT_CELL_SIZE_M = 0.25
DEFAULT_IMAGE_GRID_SIZE = 8
SUPPORTED_SENSOR_TYPES = {
    "monocular_camera", "stereo_camera", "depth_camera", "lidar_2d",
    "lidar_3d", "radar", "sonar", "gps", "wheel_odometry", "imu", "other",
}
EARTH_RADIUS_M = 6_371_000.0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _norm_sensor_type(value: Any) -> str:
    raw = str(value or "other").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "camera": "monocular_camera", "rgb_camera": "monocular_camera",
        "stereo": "stereo_camera", "depth": "depth_camera", "lidar": "lidar_2d",
        "2d_lidar": "lidar_2d", "3d_lidar": "lidar_3d", "pointcloud": "lidar_3d",
        "point_cloud": "lidar_3d", "gps_gnss": "gps", "odometry": "wheel_odometry",
    }
    raw = aliases.get(raw, raw)
    return raw if raw in SUPPORTED_SENSOR_TYPES else "other"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("sensor frame JSON root must be an object")
    return value


def load_sensor_frames(sensor_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Read normalized sensor-frame JSON files. Invalid files become warnings, not crashes."""
    frames: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    sensor_dir = sensor_dir.expanduser()
    if not sensor_dir.exists():
        return frames, errors
    for path in sorted(sensor_dir.rglob("*.json")):
        try:
            frame = _read_json(path)
            sensor_type = _norm_sensor_type(frame.get("sensor_type"))
            frame_id = str(frame.get("frame_id") or path.stem)[:160]
            frames.append({**frame, "sensor_type": sensor_type, "frame_id": frame_id, "_path": str(path.resolve())})
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            errors.append({"file": str(path), "error": str(exc)[:300]})
            LOGGER.warning("Ignoring invalid sensor frame %s: %s", path, exc)
    return frames, errors


def _normalized_image_grid(observation: dict[str, Any], size: int = DEFAULT_IMAGE_GRID_SIZE) -> dict[str, Any]:
    """Rasterize normalized detector boxes into an image-local grid, never a metric map."""
    cells: list[list[list[int]]] = [[[] for _ in range(size)] for _ in range(size)]
    objects = observation.get("observed_objects", [])
    normalized_objects: list[dict[str, Any]] = []
    if not isinstance(objects, list):
        objects = []

    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            continue
        box = obj.get("bounding_box_normalized")
        if not isinstance(box, list) or len(box) != 4:
            continue
        coords = [_as_float(v) for v in box]
        if any(v is None or v < 0.0 or v > 1.0 for v in coords):
            continue
        x0, y0, x1, y1 = (float(v) for v in coords)
        if x1 < x0 or y1 < y0:
            continue
        # Keep zero-width/height valid boxes visible by assigning their center cell.
        col_start = min(size - 1, int(x0 * size))
        col_end = min(size - 1, int(min(0.999999, x1) * size))
        row_start = min(size - 1, int(y0 * size))
        row_end = min(size - 1, int(min(0.999999, y1) * size))
        for row in range(row_start, row_end + 1):
            for col in range(col_start, col_end + 1):
                cells[row][col].append(index)
        normalized_objects.append({
            "object_index": index,
            "label": str(obj.get("label") or "unlabelled object")[:120],
            "confidence": _as_float(obj.get("confidence")),
            "bounding_box_normalized": [round(x0, 5), round(y0, 5), round(x1, 5), round(y1, 5)],
            "center_u": round((x0 + x1) / 2.0, 5),
            "center_v": round((y0 + y1) / 2.0, 5),
        })

    # X means one or more detector boxes intersect this image cell. . means no box
    # was assigned; it emphatically does not mean the underlying ground is free.
    symbols = ["".join("X" if cells[r][c] else "." for c in range(size)) for r in range(size)]
    label_by_original_index = {obj["object_index"]: obj["label"] for obj in normalized_objects}
    cell_objects = [
        {"row": r, "column": c, "object_indices": cells[r][c],
         "labels": list(dict.fromkeys(label_by_original_index[i] for i in cells[r][c] if i in label_by_original_index))}
        for r in range(size) for c in range(size) if cells[r][c]
    ]
    return {
        "grid_size_rows": size,
        "grid_size_columns": size,
        "coordinate_system": "normalized_image_coordinates_not_metric",
        "origin": "top_left_of_image",
        "symbols": symbols,
        "symbol_legend": {"X": "one or more visual detection boxes overlap this image cell", ".": "no detection box assigned; not evidence of obstacle-free ground"},
        "detected_object_count_with_boxes": len(normalized_objects),
        "objects": normalized_objects,
        "cells_with_detections": cell_objects,
    }


def _pose(frame: dict[str, Any]) -> dict[str, float] | None:
    pose = frame.get("pose")
    if not isinstance(pose, dict):
        return None
    x = _as_float(pose.get("x_m"))
    y = _as_float(pose.get("y_m"))
    yaw = _as_float(pose.get("yaw_deg"))
    if x is None or y is None or yaw is None:
        return None
    z = _as_float(pose.get("z_m")) or 0.0
    pitch = _as_float(pose.get("pitch_deg")) or 0.0
    roll = _as_float(pose.get("roll_deg")) or 0.0
    pos_std = _as_float(pose.get("position_stddev_m"))
    yaw_std = _as_float(pose.get("yaw_stddev_deg"))
    return {"x_m": x, "y_m": y, "z_m": z, "yaw_deg": yaw,
            "pitch_deg": pitch, "roll_deg": roll,
            "position_stddev_m": pos_std, "yaw_stddev_deg": yaw_std,
            "quality": str(pose.get("quality") or "unspecified"),
            "source": str(pose.get("source") or "unspecified")}


def _transform_local_xy(x_forward: float, y_left: float, pose: dict[str, Any]) -> tuple[float, float]:
    """Project a level 2-D sensor ray into the map frame using pose yaw."""
    yaw = math.radians(float(pose["yaw_deg"]))
    return (
        float(pose["x_m"]) + x_forward * math.cos(yaw) - y_left * math.sin(yaw),
        float(pose["y_m"]) + x_forward * math.sin(yaw) + y_left * math.cos(yaw),
    )


def _transform_local_xyz(x_forward: float, y_left: float, z_up: float, pose: dict[str, Any]) -> tuple[float, float, float]:
    """Transform sensor-local XYZ to map XYZ using roll, pitch (nose-down positive), yaw."""
    roll = math.radians(float(pose.get("roll_deg", 0.0)))
    pitch = math.radians(float(pose.get("pitch_deg", 0.0)))
    yaw = math.radians(float(pose["yaw_deg"]))
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    # R = Rz(yaw) * Ry(pitch) * Rx(roll); axes are x-forward/y-left/z-up.
    x_rot = cp * x_forward + sp * sr * y_left + sp * cr * z_up
    y_rot = cr * y_left - sr * z_up
    z_rot = -sp * x_forward + cp * sr * y_left + cp * cr * z_up
    x_map = cy * x_rot - sy * y_rot
    y_map = sy * x_rot + cy * y_rot
    z_map = z_rot
    return (float(pose["x_m"]) + x_map, float(pose["y_m"]) + y_map, float(pose.get("z_m", 0.0)) + z_map)


def _cell_index(x_m: float, y_m: float, cell_size_m: float) -> tuple[int, int]:
    return math.floor(x_m / cell_size_m), math.floor(y_m / cell_size_m)


def _bresenham(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    x0, y0 = start
    x1, y1 = end
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    cells: list[tuple[int, int]] = []
    while True:
        cells.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return cells


def _new_cell() -> dict[str, Any]:
    return {"occupancy_log_odds": 0.0, "free_ray_count": 0, "hit_count": 0,
            "ground_return_count": 0, "unclassified_return_count": 0,
            "coverage_count": 0, "gps_marker_count": 0, "survey_photo_ids": [],
            "evidence_sources": [], "object_labels": []}


def _ensure_cell(cells: dict[tuple[int, int], dict[str, Any]], key: tuple[int, int]) -> dict[str, Any]:
    return cells.setdefault(key, _new_cell())


def _append_limited(items: list[str], value: str, cap: int = 12) -> None:
    if value and value not in items and len(items) < cap:
        items.append(value)


def _add_ray(cells: dict[tuple[int, int], dict[str, Any]], pose: dict[str, Any], angle_deg: float,
             distance_m: float, cell_size_m: float, *, hit: bool, source_id: str) -> None:
    if not math.isfinite(distance_m) or distance_m <= 0:
        return
    angle = math.radians(angle_deg)
    local_x, local_y = distance_m * math.cos(angle), distance_m * math.sin(angle)
    end_x, end_y, _ = _transform_local_xyz(local_x, local_y, 0.0, pose)
    start_cell = _cell_index(float(pose["x_m"]), float(pose["y_m"]), cell_size_m)
    end_cell = _cell_index(end_x, end_y, cell_size_m)
    ray = _bresenham(start_cell, end_cell)
    for cell_key in ray[:-1] if hit else ray:
        cell = _ensure_cell(cells, cell_key)
        cell["occupancy_log_odds"] = max(-5.0, cell["occupancy_log_odds"] - 0.35)
        cell["free_ray_count"] += 1
        _append_limited(cell["evidence_sources"], source_id)
    if hit:
        cell = _ensure_cell(cells, end_cell)
        cell["occupancy_log_odds"] = min(5.0, cell["occupancy_log_odds"] + 0.85)
        cell["hit_count"] += 1
        _append_limited(cell["evidence_sources"], source_id)


def _process_range_scan(frame: dict[str, Any], pose: dict[str, Any], cells: dict[tuple[int, int], dict[str, Any]], cell_size_m: float) -> dict[str, Any]:
    scan = frame.get("range_scan")
    if not isinstance(scan, dict):
        return {"valid_sample_count": 0, "error": "missing range_scan object"}
    ranges = scan.get("ranges_m")
    amin = _as_float(scan.get("angle_min_deg"))
    inc = _as_float(scan.get("angle_increment_deg"))
    rmin = _as_float(scan.get("range_min_m")) or 0.0
    rmax = _as_float(scan.get("range_max_m"))
    if not isinstance(ranges, list) or amin is None or inc is None:
        return {"valid_sample_count": 0, "error": "range_scan requires ranges_m, angle_min_deg and angle_increment_deg"}
    valid_returns: list[float] = []
    max_observed_range = rmax or 0.0
    sensor_id = str(frame.get("frame_id", "sensor_frame"))
    for index, raw_range in enumerate(ranges):
        distance = _as_float(raw_range)
        if distance is None or distance <= 0 or distance < rmin:
            continue
        angle = amin + index * inc
        if rmax is not None and distance >= rmax:
            # Max-range/no-return value: ray indicates visibility up to that range,
            # but do not declare the endpoint occupied.
            _add_ray(cells, pose, angle, rmax, cell_size_m, hit=False, source_id=sensor_id)
            max_observed_range = max(max_observed_range, rmax)
            continue
        valid_returns.append(distance)
        _add_ray(cells, pose, angle, distance, cell_size_m, hit=True, source_id=sensor_id)
    return {
        "valid_sample_count": len(valid_returns),
        "sample_count": len(ranges),
        "min_return_m": round(min(valid_returns), 3) if valid_returns else None,
        "max_return_m": round(max(valid_returns), 3) if valid_returns else None,
        "median_return_m": round(statistics.median(valid_returns), 3) if valid_returns else None,
        "configured_max_range_m": rmax,
        "max_ray_extent_m": round(max_observed_range, 3) if max_observed_range else None,
        "range_basis": "range_scan_measurements",
    }


def _process_points(frame: dict[str, Any], pose: dict[str, Any], cells: dict[tuple[int, int], dict[str, Any]], cell_size_m: float) -> dict[str, Any]:
    points = frame.get("points")
    if not isinstance(points, list):
        return {"valid_sample_count": 0, "error": "missing points array"}
    valid_distances: list[float] = []
    sensor_id = str(frame.get("frame_id", "sensor_frame"))
    start_cell = _cell_index(float(pose["x_m"]), float(pose["y_m"]), cell_size_m)
    ground_plane_z = _as_float(frame.get("ground_plane_z_m"))
    obstacle_height_min = _as_float(frame.get("obstacle_height_min_m"))
    if obstacle_height_min is None:
        obstacle_height_min = 0.08
    obstacle_height_max = _as_float(frame.get("obstacle_height_max_m"))
    if obstacle_height_max is None:
        obstacle_height_max = 2.5
    ground_returns = 0
    obstacle_returns = 0
    unclassified_returns = 0

    for point in points:
        if not isinstance(point, dict):
            continue
        x = _as_float(point.get("x_m"))
        y = _as_float(point.get("y_m"))
        z = _as_float(point.get("z_m")) or 0.0
        if x is None or y is None:
            continue
        distance = math.sqrt(x * x + y * y + z * z)
        if distance <= 0 or not math.isfinite(distance):
            continue
        valid_distances.append(distance)
        # Point cloud coordinates are sensor-aligned x-forward/y-left/z-up.
        end_x, end_y, end_z = _transform_local_xyz(x, y, z, pose)
        end_cell = _cell_index(end_x, end_y, cell_size_m)
        ray = _bresenham(start_cell, end_cell)
        for key in ray[:-1]:
            cell = _ensure_cell(cells, key)
            cell["occupancy_log_odds"] = max(-5.0, cell["occupancy_log_odds"] - 0.20)
            cell["free_ray_count"] += 1
            _append_limited(cell["evidence_sources"], sensor_id)

        cell = _ensure_cell(cells, end_cell)
        cell.setdefault("point_z_values_m", []).append(round(end_z, 3))
        _append_limited(cell["evidence_sources"], sensor_id)
        classification = str(point.get("classification", point.get("label", ""))).strip().lower()
        is_ground = point.get("is_ground") is True or classification in {"ground", "floor", "terrain_surface"}
        is_obstacle = point.get("is_obstacle") is True or classification in {"obstacle", "rock", "boulder", "object"}
        height_above_ground = (z - ground_plane_z) if ground_plane_z is not None else None

        if is_ground or (ground_plane_z is not None and height_above_ground is not None and height_above_ground < obstacle_height_min):
            cell["ground_return_count"] += 1
            ground_returns += 1
        elif is_obstacle or (height_above_ground is not None and obstacle_height_min <= height_above_ground <= obstacle_height_max):
            cell["occupancy_log_odds"] = min(5.0, cell["occupancy_log_odds"] + 0.75)
            cell["hit_count"] += 1
            obstacle_returns += 1
        else:
            # For 3-D LiDAR/depth/stereo data without ground segmentation or a
            # supplied ground-plane reference, the return's height class is unknown.
            # Keep the return spatially represented without pretending it is an obstacle.
            cell["unclassified_return_count"] += 1
            unclassified_returns += 1

    return {
        "valid_sample_count": len(valid_distances),
        "min_return_m": round(min(valid_distances), 3) if valid_distances else None,
        "max_return_m": round(max(valid_distances), 3) if valid_distances else None,
        "median_return_m": round(statistics.median(valid_distances), 3) if valid_distances else None,
        "ground_return_count": ground_returns,
        "obstacle_return_count": obstacle_returns,
        "unclassified_return_count": unclassified_returns,
        "ground_plane_reference_supplied": ground_plane_z is not None,
        "range_basis": "point_cloud_euclidean_range",
    }


def _camera_coverage(frame: dict[str, Any], pose: dict[str, Any], cells: dict[tuple[int, int], dict[str, Any]], cell_size_m: float) -> dict[str, Any]:
    camera = frame.get("camera")
    if not isinstance(camera, dict):
        return {"available": False, "reason": "camera intrinsics, camera height and pitch were not supplied"}
    numeric_fields = ("fx_px", "fy_px", "cx_px", "cy_px", "image_width_px", "image_height_px", "camera_height_m", "pitch_down_deg")
    values = {key: _as_float(camera.get(key)) for key in numeric_fields}
    if any(value is None for value in values.values()):
        return {"available": False, "reason": "camera coverage needs fx/fy/cx/cy, image dimensions, camera height, and downward pitch calibration"}
    fx, fy = values["fx_px"], values["fy_px"]
    cx, cy = values["cx_px"], values["cy_px"]
    width, height = int(values["image_width_px"]), int(values["image_height_px"])
    camera_height, pitch_down = values["camera_height_m"], values["pitch_down_deg"]
    if fx <= 0 or fy <= 0 or width <= 0 or height <= 0 or camera_height <= 0:
        return {"available": False, "reason": "camera calibration values must be positive"}
    # This is an approximate footprint under a locally flat-ground assumption.
    # It records coverage only and does not mark cells free or obstacle-free.
    samples: list[tuple[float, float]] = []
    row_fractions = (0.55, 0.65, 0.75, 0.85, 0.95)
    col_fractions = (0.05, 0.25, 0.50, 0.75, 0.95)
    frame_id = str(frame.get("frame_id", "camera_frame"))
    for rf in row_fractions:
        v = min(height - 1.0, rf * (height - 1.0))
        pixel_down_deg = math.degrees(math.atan((v - cy) / fy))
        depression_deg = pitch_down + pixel_down_deg
        if depression_deg <= 1.0 or depression_deg >= 89.5:
            continue
        forward = camera_height / math.tan(math.radians(depression_deg))
        if not math.isfinite(forward) or forward <= 0 or forward > 150.0:
            continue
        for cf in col_fractions:
            u = min(width - 1.0, cf * (width - 1.0))
            right_angle = math.atan((u - cx) / fx)
            # Camera image-right corresponds to rover-right, i.e. negative y-left.
            x_forward = forward
            y_left = -forward * math.tan(right_angle)
            map_x, map_y = _transform_local_xy(x_forward, y_left, pose)
            key = _cell_index(map_x, map_y, cell_size_m)
            cell = _ensure_cell(cells, key)
            cell["coverage_count"] += 1
            _append_limited(cell["evidence_sources"], frame_id)
            samples.append((x_forward, y_left))
    if not samples:
        return {"available": False, "reason": "no camera rays intersected the estimated flat ground in the sampled view"}
    ground_forward = [x for x, _ in samples]
    ground_abs_y = [abs(y) for _, y in samples]
    max_ground_range = max(math.hypot(x, y) for x, y in samples)
    return {
        "available": True,
        "range_basis": "calibrated_camera_ground_plane_assumption",
        "assumption": "locally flat ground; terrain relief, occlusion and camera calibration error can invalidate this estimate",
        "sampled_ground_intersections": len(samples),
        "min_forward_intersection_m": round(min(ground_forward), 2),
        "max_forward_intersection_m": round(max(ground_forward), 2),
        "max_sampled_slant_range_m": round(max_ground_range, 2),
        "max_half_width_m": round(max(ground_abs_y), 2),
        "occupancy_inferred": False,
    }


def _gps_to_local_xy(latitude: float, longitude: float, origin_lat: float, origin_lon: float) -> tuple[float, float]:
    lat0 = math.radians(origin_lat)
    x_east = EARTH_RADIUS_M * math.radians(longitude - origin_lon) * math.cos(lat0)
    y_north = EARTH_RADIUS_M * math.radians(latitude - origin_lat)
    return x_east, y_north


def _cell_status(cell: dict[str, Any]) -> str:
    if cell["occupancy_log_odds"] >= 0.7:
        return "occupied_candidate"
    if cell["hit_count"] > 0:
        return "return_conflicting_evidence"
    if cell.get("unclassified_return_count", 0) > 0:
        return "return_unclassified"
    if cell.get("ground_return_count", 0) > 0:
        return "ground_surface_observed"
    if cell["occupancy_log_odds"] <= -0.2 and cell["free_ray_count"] > 0:
        return "ray_observed_no_return"
    if cell["coverage_count"] > 0:
        return "camera_coverage_only"
    if cell["survey_photo_ids"] or cell.get("gps_marker_count", 0) > 0:
        return "survey_location_only"
    return "unknown"


def build_map_snapshot(
    observations: list[dict[str, Any]],
    *,
    sensor_dir: Path = Path("data/sensor_frames"),
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    image_grid_size: int = DEFAULT_IMAGE_GRID_SIZE,
) -> dict[str, Any]:
    """Build a two-layer (image-local + metric) sparse map snapshot from current evidence."""
    if not math.isfinite(cell_size_m) or not 0.02 <= cell_size_m <= 5.0:
        raise ValueError("grid cell size must be between 0.02 and 5.0 metres")
    if not 2 <= image_grid_size <= 32:
        raise ValueError("image grid size must be between 2 and 32 cells per side")

    image_maps: list[dict[str, Any]] = []
    for obs in observations:
        image_maps.append({
            "photo_id": obs.get("photo_id"),
            "image_sha256": obs.get("image_sha256"),
            "source_name": Path(str(obs.get("source_path", "unknown"))).name,
            "grid": _normalized_image_grid(obs, image_grid_size),
            "metric_projection_available": False,
            "metric_projection_reason": "No per-image camera calibration plus trusted pose/depth association was attached to this observation.",
        })

    frames, frame_errors = load_sensor_frames(sensor_dir)
    cells: dict[tuple[int, int], dict[str, Any]] = {}
    sensor_stats: dict[str, dict[str, Any]] = {}
    metric_frame_count = 0
    localization_missing_frames = 0
    frame_results: list[dict[str, Any]] = []
    camera_coverage: list[dict[str, Any]] = []
    source_by_photo: dict[str, list[str]] = defaultdict(list)

    # GPS positions are useful survey-location markers. Their local ENU origin is
    # the first provided fix, never an origin guessed from pixels. Sensor-frame GPS
    # fixes may be written as top-level latitude/longitude or inside gps_fix.
    gps_observations = [o for o in observations if _as_float(o.get("latitude")) is not None and _as_float(o.get("longitude")) is not None]
    gps_frames: list[dict[str, Any]] = []
    for frame in frames:
        fix = frame.get("gps_fix") if isinstance(frame.get("gps_fix"), dict) else frame
        lat, lon = _as_float(fix.get("latitude")), _as_float(fix.get("longitude"))
        if frame.get("sensor_type") == "gps" and lat is not None and lon is not None:
            gps_frames.append({**frame, "_latitude": lat, "_longitude": lon})
    gps_origin = None
    if gps_observations:
        gps_origin = (float(gps_observations[0]["latitude"]), float(gps_observations[0]["longitude"]))
    elif gps_frames:
        gps_origin = (float(gps_frames[0]["_latitude"]), float(gps_frames[0]["_longitude"]))

    if gps_origin:
        for obs in gps_observations:
            east, north = _gps_to_local_xy(float(obs["latitude"]), float(obs["longitude"]), *gps_origin)
            key = _cell_index(east, north, cell_size_m)
            cell = _ensure_cell(cells, key)
            _append_limited(cell["survey_photo_ids"], str(obs.get("photo_id", "")))
            _append_limited(cell["evidence_sources"], "gps_tagged_image")
        for frame in gps_frames:
            east, north = _gps_to_local_xy(float(frame["_latitude"]), float(frame["_longitude"]), *gps_origin)
            key = _cell_index(east, north, cell_size_m)
            cell = _ensure_cell(cells, key)
            photo_id = str(frame.get("photo_id") or "")
            if photo_id:
                _append_limited(cell["survey_photo_ids"], photo_id)
            cell["gps_marker_count"] += 1
            _append_limited(cell["evidence_sources"], f"gps:{frame['frame_id']}")

    for frame in frames:
        sensor_type = frame["sensor_type"]
        pose = _pose(frame)
        if sensor_type == "gps" and isinstance(frame.get("gps_fix"), dict) or sensor_type == "gps" and _as_float(frame.get("latitude")) is not None:
            fix = frame.get("gps_fix") if isinstance(frame.get("gps_fix"), dict) else frame
            lat, lon = _as_float(fix.get("latitude")), _as_float(fix.get("longitude"))
            fix_valid = lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180
            frame_result = {
                "frame_id": frame["frame_id"], "sensor_type": sensor_type,
                "photo_id": frame.get("photo_id"), "timestamp_utc": frame.get("timestamp_utc"),
                "source_file": Path(frame["_path"]).name, "pose_available": False,
                "gps_fix_available": bool(fix_valid),
                "mapping_status": "gps_location_marker_only" if fix_valid else "gps_fix_invalid_or_missing",
            }
            stats = sensor_stats.setdefault(sensor_type, {
                "frame_count": 0, "localized_frame_count": 0, "valid_sample_count": 0,
                "max_observed_range_m": None, "max_ray_extent_m": None,
                "min_return_m": None, "configured_max_range_m": None, "range_bases": [],
            })
            stats["frame_count"] += 1
            if fix_valid and gps_origin:
                east, north = _gps_to_local_xy(lat, lon, *gps_origin)
                frame_result["map_position_m"] = {"x_m": round(east, 3), "y_m": round(north, 3)}
            else:
                localization_missing_frames += 1
            frame_result["range_summary"] = {"valid_sample_count": 0, "range_basis": "gps_has_position_not_obstacle_range"}
            frame_results.append(frame_result)
            continue

        frame_result: dict[str, Any] = {
            "frame_id": frame["frame_id"],
            "sensor_type": sensor_type,
            "photo_id": frame.get("photo_id"),
            "timestamp_utc": frame.get("timestamp_utc"),
            "source_file": Path(frame["_path"]).name,
            "pose_available": pose is not None,
            "pose_quality": pose.get("quality") if pose else "unavailable",
            "pose_source": pose.get("source") if pose else None,
            "pose_position_stddev_m": pose.get("position_stddev_m") if pose else None,
            "pose_yaw_stddev_deg": pose.get("yaw_stddev_deg") if pose else None,
            "mapping_status": "localized" if pose else "range_data_only_unlocalized",
        }
        stats = sensor_stats.setdefault(sensor_type, {
            "frame_count": 0, "localized_frame_count": 0, "valid_sample_count": 0,
            "max_observed_range_m": None, "max_ray_extent_m": None,
            "min_return_m": None, "configured_max_range_m": None,
            "range_bases": [],
        })
        stats["frame_count"] += 1
        if pose is None:
            localization_missing_frames += 1
            scan = frame.get("range_scan") if isinstance(frame.get("range_scan"), dict) else {}
            pts = frame.get("points") if isinstance(frame.get("points"), list) else []
            distances: list[float] = []
            if isinstance(scan.get("ranges_m"), list):
                rmax = _as_float(scan.get("range_max_m"))
                stats["configured_max_range_m"] = max(stats["configured_max_range_m"] or 0, rmax or 0) or None
                for raw in scan["ranges_m"]:
                    r = _as_float(raw)
                    rmin = _as_float(scan.get("range_min_m")) or 0.0
                    if r is not None and r > 0 and r >= rmin and (rmax is None or r < rmax):
                        distances.append(r)
            for p in pts:
                if isinstance(p, dict):
                    x, y = _as_float(p.get("x_m")), _as_float(p.get("y_m"))
                    if x is not None and y is not None:
                        distances.append(math.hypot(x, y))
            frame_result["range_summary"] = {
                "valid_sample_count": len(distances),
                "min_return_m": round(min(distances), 3) if distances else None,
                "max_return_m": round(max(distances), 3) if distances else None,
                "max_ray_extent_m": rmax if isinstance(scan, dict) and _as_float(scan.get("range_max_m")) is not None else (round(max(distances), 3) if distances else None),
                "configured_max_range_m": _as_float(scan.get("range_max_m")) if isinstance(scan, dict) else None,
                "range_basis": "measurements_without_map_pose",
            }
            range_summary = frame_result["range_summary"]
        else:
            metric_frame_count += 1
            stats["localized_frame_count"] += 1
            if "range_scan" in frame:
                range_summary = _process_range_scan(frame, pose, cells, cell_size_m)
            elif "points" in frame:
                range_summary = _process_points(frame, pose, cells, cell_size_m)
            else:
                range_summary = {"valid_sample_count": 0, "error": "no range_scan or points measurements"}
            if sensor_type in {"monocular_camera", "stereo_camera", "depth_camera"} and isinstance(frame.get("camera"), dict):
                coverage = _camera_coverage(frame, pose, cells, cell_size_m)
                coverage_record = {"frame_id": frame["frame_id"], "photo_id": frame.get("photo_id"), **coverage}
                camera_coverage.append(coverage_record)
                frame_result["camera_ground_coverage"] = coverage
        frame_result["range_summary"] = range_summary
        stats["valid_sample_count"] += int(range_summary.get("valid_sample_count", 0) or 0)
        maximum = _as_float(range_summary.get("max_return_m"))
        ray_extent = _as_float(range_summary.get("max_ray_extent_m"))
        minimum = _as_float(range_summary.get("min_return_m"))
        config_max = _as_float(range_summary.get("configured_max_range_m"))
        if maximum is not None:
            stats["max_observed_range_m"] = max(stats["max_observed_range_m"] or 0.0, maximum)
        if ray_extent is not None:
            stats["max_ray_extent_m"] = max(stats["max_ray_extent_m"] or 0.0, ray_extent)
        if minimum is not None:
            stats["min_return_m"] = min(stats["min_return_m"] or minimum, minimum)
        if config_max is not None:
            stats["configured_max_range_m"] = max(stats["configured_max_range_m"] or 0.0, config_max)
        _append_limited(stats["range_bases"], str(range_summary.get("range_basis", "unknown")))
        if frame.get("photo_id"):
            source_by_photo[str(frame["photo_id"])].append(str(frame["frame_id"]))
        frame_results.append(frame_result)

    for frame in frames:
        photo_id = str(frame.get("photo_id") or "")
        if photo_id:
            source_by_photo[photo_id].append(str(frame.get("frame_id", "")))
    for image_map in image_maps:
        associated = list(dict.fromkeys(source_by_photo.get(str(image_map.get("photo_id") or ""), [])))
        image_map["associated_sensor_frame_ids"] = associated
        if associated:
            image_map["metric_projection_reason"] = (
                "Sensor frames are associated by photo_id, but object bounding boxes are not projected into metric space unless per-object depth/pixel correspondence is supplied."
            )

    serialized_cells: list[dict[str, Any]] = []
    for (gx, gy), cell in sorted(cells.items(), key=lambda item: (item[0][1], item[0][0])):
        hit_count = cell["hit_count"]
        free_count = cell["free_ray_count"]
        status = _cell_status(cell)
        serialized_cells.append({
            "grid_x": gx, "grid_y": gy,
            "center_x_m": round((gx + 0.5) * cell_size_m, 3),
            "center_y_m": round((gy + 0.5) * cell_size_m, 3),
            "occupancy_log_odds": round(cell["occupancy_log_odds"], 3),
            "status": status,
            "hit_count": hit_count,
            "free_ray_count": free_count,
            "ground_return_count": cell.get("ground_return_count", 0),
            "unclassified_return_count": cell.get("unclassified_return_count", 0),
            "coverage_count": cell["coverage_count"],
            "gps_marker_count": cell.get("gps_marker_count", 0),
            "survey_photo_ids": cell["survey_photo_ids"],
            "evidence_sources": cell["evidence_sources"],
            "object_labels": cell["object_labels"],
            "point_z_values_m": cell.get("point_z_values_m", [])[:32],
        })

    range_summary = {
        sensor_type: {
            **stats,
            "max_observed_range_m": round(stats["max_observed_range_m"], 3) if stats["max_observed_range_m"] is not None else None,
            "max_ray_extent_m": round(stats["max_ray_extent_m"], 3) if stats["max_ray_extent_m"] is not None else None,
            "min_return_m": round(stats["min_return_m"], 3) if stats["min_return_m"] is not None else None,
            "configured_max_range_m": round(stats["configured_max_range_m"], 3) if stats["configured_max_range_m"] is not None else None,
        }
        for sensor_type, stats in sorted(sensor_stats.items())
    }
    occupancy_cell_count = sum(1 for c in serialized_cells if c["status"] in {"occupied_candidate", "ray_observed_no_return"})
    point_return_cell_count = sum(1 for c in serialized_cells if c["status"] in {"ground_surface_observed", "return_unclassified", "return_conflicting_evidence"})
    coverage_cell_count = sum(1 for c in serialized_cells if c["status"] == "camera_coverage_only")
    if occupancy_cell_count or point_return_cell_count:
        status = "metric_sensor_map_available"
    elif coverage_cell_count:
        status = "camera_ground_coverage_available"
    elif gps_observations or gps_frames:
        status = "gps_survey_locations_only"
    else:
        status = "image_relative_grid_only"

    snapshot = {
        "schema_version": 1,
        "generated_at_utc": utc_now(),
        "mapping_status": status,
        "grid": {
            "cell_size_m": cell_size_m,
            "coordinate_frame": "shared 2-D metric map frame; sensor pose x_m/y_m/yaw_deg must share this frame. With GPS anchoring, origin is the first georeferenced observation and axes are local east/north; otherwise it is a software-defined local frame.",
            "axis_convention": "sensor-local points use x forward, y left, z up; pose x/y live in the shared fixed map frame; yaw is counter-clockwise from map +x",
            "occupancy_model": "sparse log-odds from range rays/point returns; camera-only ground intersections add coverage, never free-space evidence",
            "cell_count": len(serialized_cells),
            "occupancy_or_ray_cells": occupancy_cell_count,
            "point_return_cells": point_return_cell_count,
            "camera_coverage_only_cells": coverage_cell_count,
            "cells": serialized_cells,
            "gps_origin_latitude_longitude": list(gps_origin) if gps_origin else None,
        },
        "sensor_summary": {
            "sensor_frames_found": len(frames),
            "localized_sensor_frames": metric_frame_count,
            "unlocalized_sensor_frames": localization_missing_frames,
            "sensor_types": sorted(sensor_stats),
            "range_by_sensor_type": range_summary,
            "camera_ground_coverage": camera_coverage,
            "range_estimation_note": "max_observed_range_m is the farthest valid measured return; max_ray_extent_m is scan coverage including explicit max-range/no-return rays; configured_max_range_m is sensor metadata, not a guarantee of reliable range on every surface.",
            "ordinary_rgb_note": "An ordinary monocular RGB photo and normalized GLM boxes provide no trustworthy absolute range without calibration plus geometry/scale; no metres are fabricated.",
        },
        "image_relative_grids": image_maps,
        "frames": frame_results,
        "input_errors": frame_errors,
        "coverage_gaps": [],
        "notes": [
            "Image-relative grids show detection-box locations inside each image only; different photos are not aligned in space unless pose or georeferencing exists.",
            "A grid cell with no detection is not proven free or traversable.",
            "3-D point returns are not automatically classified as obstacles: without point classification or a supplied ground_plane_z_m, they remain return_unclassified.",
            "Metric map cells remain unknown outside measured rays/points/coverage; no unseen cells are filled as free.",
            "Map fusion currently records pose quality/source but does not run SLAM or probabilistically compensate for pose drift; check odometry/IMU/GNSS uncertainty before using the grid for navigation.",
            "Map is an evidence product, not a certified navigation map. It does not command motors or prove rover safety.",
        ],
    }
    if not frames:
        snapshot["coverage_gaps"].append("No normalized sensor-frame files found. Only image-relative grids can be built from current survey records.")
    if localization_missing_frames:
        snapshot["coverage_gaps"].append(f"{localization_missing_frames} sensor frame(s) have measurements but no x_m/y_m/yaw_deg pose, so they cannot be fused into the shared metric grid.")
    if not gps_observations and not metric_frame_count:
        snapshot["coverage_gaps"].append("No trusted global/local pose is available to align photos or sensor measurements into a shared metric map.")
    if observations and frames and not any(frame.get("photo_id") for frame in frames):
        snapshot["coverage_gaps"].append("Sensor frames are not associated with photo_id values; image-to-sensor fusion is not available for current observations.")
    return snapshot


def compact_map_summary(snapshot: dict[str, Any], *, max_items: int = 20) -> dict[str, Any]:
    """Return a bounded summary suitable for a mission-planner prompt and CLI report."""
    grid = snapshot.get("grid", {})
    sensors = snapshot.get("sensor_summary", {})
    ranges = sensors.get("range_by_sensor_type", {})
    previews = []
    for image_map in (snapshot.get("image_relative_grids", []) or [])[:max_items]:
        g = image_map.get("grid", {}) if isinstance(image_map, dict) else {}
        previews.append({
            "photo_id": image_map.get("photo_id"),
            "grid_size": f"{g.get('grid_size_rows', 0)}x{g.get('grid_size_columns', 0)}",
            "symbols": g.get("symbols", []),
            "detected_object_count_with_boxes": g.get("detected_object_count_with_boxes", 0),
            "metric_projection_available": image_map.get("metric_projection_available", False),
        })
    cell_preview = []
    for cell in (grid.get("cells", []) or [])[:100]:
        cell_preview.append({
            "grid_x": cell.get("grid_x"), "grid_y": cell.get("grid_y"),
            "center_x_m": cell.get("center_x_m"), "center_y_m": cell.get("center_y_m"),
            "status": cell.get("status"), "hit_count": cell.get("hit_count"),
            "free_ray_count": cell.get("free_ray_count"), "coverage_count": cell.get("coverage_count"),
        })
    return {
        "mapping_status": snapshot.get("mapping_status", "unknown"),
        "grid_cell_size_m": grid.get("cell_size_m"),
        "metric_grid_cell_count": grid.get("cell_count", 0),
        "occupied_or_ray_cells": grid.get("occupancy_or_ray_cells", 0),
        "point_return_cells": grid.get("point_return_cells", 0),
        "camera_coverage_only_cells": grid.get("camera_coverage_only_cells", 0),
        "metric_grid_preview_first_100_cells": cell_preview,
        "image_relative_grid_count": len(snapshot.get("image_relative_grids", [])),
        "image_grid_previews": previews,
        "sensor_frames_found": sensors.get("sensor_frames_found", 0),
        "localized_sensor_frames": sensors.get("localized_sensor_frames", 0),
        "sensor_types": sensors.get("sensor_types", []),
        "range_by_sensor_type": ranges,
        "camera_ground_coverage": (sensors.get("camera_ground_coverage", []) or [])[:max_items],
        "coverage_gaps": (snapshot.get("coverage_gaps", []) or [])[:max_items],
        "range_estimation_note": sensors.get("range_estimation_note", ""),
        "ordinary_rgb_note": sensors.get("ordinary_rgb_note", ""),
    }


def render_metric_grid_ascii(snapshot: dict[str, Any], max_side: int = 100) -> str:
    """Render sparse metric cells as a small text grid; blank means unobserved/unknown."""
    grid = snapshot.get("grid", {})
    cells = grid.get("cells", [])
    if not cells:
        return (
            "No metric world grid is available yet.\n"
            "Image-relative grids exist in latest_map.json, but images are not spatially aligned.\n"
        )
    xs = [int(c["grid_x"]) for c in cells]
    ys = [int(c["grid_y"]) for c in cells]
    xmin, xmax, ymin, ymax = min(xs), max(xs), min(ys), max(ys)
    if xmax - xmin + 1 > max_side or ymax - ymin + 1 > max_side:
        return ("Metric grid too large for ASCII display; inspect latest_map.json or use a map viewer.\n"
                f"Grid bounds: x={xmin}..{xmax}, y={ymin}..{ymax}; cell={grid.get('cell_size_m')} m.\n")
    char_by_status = {
        "occupied_candidate": "#", "ray_observed_no_return": ".",
        "camera_coverage_only": "c", "ground_surface_observed": "g",
        "return_unclassified": "u", "return_conflicting_evidence": "!",
        "survey_location_only": "P", "unknown": "?",
    }
    lookup = {(int(c["grid_x"]), int(c["grid_y"])): char_by_status.get(c["status"], "?") for c in cells}
    lines = [
        f"TerraScout metric grid (cell={grid.get('cell_size_m')} m; x={xmin}..{xmax}, y={ymin}..{ymax})",
        "View: top is larger map-y. Blank = no map evidence; . = sensor ray observed no return; # = obstacle candidate; g = ground return; u = point return not classified as ground/obstacle; ! = conflicting occupancy evidence; c = camera ground coverage only; P = GPS photo marker; ? = weak evidence.",
    ]
    for y in range(ymax, ymin - 1, -1):
        lines.append(f"{y:>5} " + "".join(lookup.get((x, y), " ") for x in range(xmin, xmax + 1)))
    lines.append("      " + "".join(str((x // 10) % 10) for x in range(xmin, xmax + 1)))
    lines.append("Metric sensor grid is not a verified safe-traversal map.")
    return "\n".join(lines) + "\n"


def render_image_grids_text(snapshot: dict[str, Any]) -> str:
    lines = [
        "TerraScout per-image normalized detection grids",
        "X = detection box overlaps cell; . = no box in cell (NOT confirmed free ground).",
        "These are image coordinates, not metric grids; different images are not spatially aligned.\n",
    ]
    for image_map in snapshot.get("image_relative_grids", []):
        grid = image_map.get("grid", {})
        lines.append(f"photo_id={image_map.get('photo_id')} ({grid.get('grid_size_rows')}x{grid.get('grid_size_columns')}); boxes={grid.get('detected_object_count_with_boxes')}")
        lines.extend(grid.get("symbols", []))
        labels = [f"{obj.get('object_index')}: {obj.get('label')} bbox={obj.get('bounding_box_normalized')}" for obj in grid.get("objects", [])]
        lines.extend(labels or ["  (no usable bounding boxes stored)"])
        lines.append("")
    return "\n".join(lines)


def save_map_snapshot(snapshot: dict[str, Any], output_dir: Path) -> Path:
    """Persist JSON map and human-readable grid exports without changing survey DB."""
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    revision_path = output_dir / f"terrain_map_{stamp}.json"
    latest_path = output_dir / "latest_map.json"
    encoded = json.dumps(snapshot, ensure_ascii=False, indent=2)
    revision_path.write_text(encoded, encoding="utf-8")
    latest_path.write_text(encoded, encoding="utf-8")
    (output_dir / "latest_metric_grid.txt").write_text(render_metric_grid_ascii(snapshot), encoding="utf-8")
    (output_dir / "latest_image_grids.txt").write_text(render_image_grids_text(snapshot), encoding="utf-8")
    return latest_path



def load_observations_from_database(database_path: Path) -> list[dict[str, Any]]:
    """Load the existing vision observations table without needing the Nebius client."""
    if not database_path.is_file():
        return []
    connection = sqlite3.connect(str(database_path), timeout=10.0)
    connection.row_factory = sqlite3.Row
    try:
        table = connection.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='observations'").fetchone()
        if table is None:
            return []
        rows = connection.execute(
            "SELECT source_path, photo_id, image_sha256, latitude, longitude, "
            "terrain_json, candidate_objects_json, uncertainties_json "
            "FROM observations ORDER BY analyzed_at_utc"
        ).fetchall()
    finally:
        connection.close()
    result: list[dict[str, Any]] = []
    for row in rows:
        try:
            terrain = json.loads(row["terrain_json"] or "{}")
            objects = json.loads(row["candidate_objects_json"] or "[]")
            uncertainties = json.loads(row["uncertainties_json"] or "[]")
        except json.JSONDecodeError:
            LOGGER.warning("Skipping malformed stored observation %s", row["photo_id"])
            continue
        result.append({
            "source_path": row["source_path"], "photo_id": row["photo_id"],
            "image_sha256": row["image_sha256"], "latitude": row["latitude"],
            "longitude": row["longitude"], "terrain": terrain if isinstance(terrain, dict) else {},
            "observed_objects": objects if isinstance(objects, list) else [],
            "uncertainties": uncertainties if isinstance(uncertainties, list) else [],
        })
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build TerraScout's image-relative and sensor-aware grid map without calling an LLM.")
    parser.add_argument("--database", type=Path, default=Path("data/terrascout.db"), help="Existing vision SQLite DB.")
    parser.add_argument("--sensor-dir", type=Path, default=Path("data/sensor_frames"), help="Directory containing normalized sensor frame JSON files.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/terrain_map"), help="Map JSON/text output directory.")
    parser.add_argument("--cell-size-m", type=float, default=DEFAULT_CELL_SIZE_M, help="Metric grid resolution in metres/cell.")
    parser.add_argument("--image-grid-size", type=int, default=DEFAULT_IMAGE_GRID_SIZE, help="Per-image normalized grid side length (2-32, default 8).")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    try:
        observations = load_observations_from_database(args.database.expanduser().resolve())
        snapshot = build_map_snapshot(
            observations,
            sensor_dir=args.sensor_dir.expanduser().resolve(),
            cell_size_m=args.cell_size_m,
            image_grid_size=args.image_grid_size,
        )
        latest_path = save_map_snapshot(snapshot, args.output_dir)
        summary = compact_map_summary(snapshot)
        print(f"Mapping status: {summary['mapping_status']}")
        print(f"Stored image-relative grids: {summary['image_relative_grid_count']}")
        print(f"Metric grid cells: {summary['metric_grid_cell_count']} (resolution {summary['grid_cell_size_m']} m/cell)")
        print(f"Sensor frames: {summary['sensor_frames_found']} total; {summary['localized_sensor_frames']} localized")
        for sensor_name, sensor in summary.get("range_by_sensor_type", {}).items():
            print(f"{sensor_name}: max return={sensor.get('max_observed_range_m')} m; max ray extent={sensor.get('max_ray_extent_m')} m; configured max={sensor.get('configured_max_range_m')} m")
        for coverage in summary.get("camera_ground_coverage", []):
            if coverage.get("available"):
                print(f"Camera {coverage.get('frame_id')}: estimated flat-ground footprint extends to {coverage.get('max_sampled_slant_range_m')} m (not measured depth)")
        for gap in summary.get("coverage_gaps", []):
            print(f"Coverage gap: {gap}")
        print(f"Full map: {latest_path.resolve()}")
        print(f"Image grids: {(args.output_dir.expanduser().resolve() / 'latest_image_grids.txt')}")
        print(f"Metric grid: {(args.output_dir.expanduser().resolve() / 'latest_metric_grid.txt')}")
    except Exception as exc:
        LOGGER.error("Mapping failed: %s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
