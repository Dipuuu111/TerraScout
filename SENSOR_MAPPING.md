# TerraScout sensor-aware grid mapping

## What Nemotron now receives

`nemotron.py` builds a mapping snapshot before asking Nemotron for the mission strategy. It writes:

- `data/terrain_map/latest_map.json` — complete sparse map snapshot, sensor range summaries, image-relative grids, gaps and mapping notes.
- `data/terrain_map/latest_image_grids.txt` — normalized per-image 8x8 detector-box grids.
- `data/terrain_map/latest_metric_grid.txt` — human-readable metric occupancy/coverage grid when measured spatial evidence is available.

The mission JSON also contains a compact mapping summary. In `--watch` mode, changes to sensor-frame JSON files trigger re-planning as well as changes to the SQLite observation database.

## Current photographs

Current SQLite image observations generally provide GLM terrain descriptions and normalized object bounding boxes. These are used to make an 8x8 **image-relative grid** for each photo. The grid answers “where in this image is a detection?” It does not answer “how many metres away is that rock?”, and separate photos are not aligned with one another until a trusted pose or georeference is supplied.

A standard monocular RGB image cannot reliably provide absolute depth by itself. A configured camera can estimate a rough ground footprint only if you supply intrinsics, camera height, downward pitch and a trusted pose. That result is marked as a flat-ground assumption and is coverage only, not free-space evidence or an obstacle position.

## Add sensor-frame data

Create `E:\TerraScout\data\sensor_frames\`. Copy `sensor_frame_template.json` to a new JSON file per reading/frame, then populate it with actual data from the hardware adapter. Do not leave placeholders or the template itself in `data/sensor_frames`.

Each normalized frame should have:

- `frame_id`: unique reading/frame ID.
- `photo_id`: optional matching photo ID to associate it with a GLM observation.
- `sensor_type`: `lidar_2d`, `lidar_3d`, `depth_camera`, `stereo_camera`, `monocular_camera`, `radar`, `sonar`, `gps`, `wheel_odometry`, `imu` or `other`.
- `pose`: sensor origin in a common fixed metric map frame. Provide `x_m`, `y_m`, and `yaw_deg`; include z/pitch/roll/quality/source when available. Sensor adapters should fuse wheel odometry/IMU/localization into this pose, rather than sending unaligned raw poses.
- measurement data: `range_scan` or `points` for range/point returns, or `camera` calibration for flat-ground footprint estimates.

### 2-D LiDAR / range scan

`range_scan.ranges_m` is an array of ranges; bearing for element i is `angle_min_deg + i * angle_increment_deg`. Set `range_max_m` to the sensor's declared maximum range. A range equal to the max is treated as max-range/no-return coverage, not an obstacle hit. The map reports farthest valid return separately from the farthest ray coverage and from the configured maximum.

### 3-D LiDAR / depth / stereo point cloud

Supply `points` as an array of `{ "x_m": ..., "y_m": ..., "z_m": ... }` in sensor-aligned coordinates: x forward, y left, z up, in metres. The `pose` places the sensor frame in the shared map frame. A future hardware adapter can convert depth images or raw point clouds into this normalized format.

**Do not label every point as a rock.** Most point clouds contain many samples of the ground. If the sensor pipeline already has segmentation, include `classification: "ground"` or `is_ground: true` for ground returns, and `classification: "rock"` or `is_obstacle: true` for confirmed obstacle candidates. Alternatively, supply `ground_plane_z_m` in the sensor-local point-cloud frame and tune `obstacle_height_min_m` (default `0.08`) / `obstacle_height_max_m` (default `2.5`) for a simple height-above-ground filter. The height filter is only a baseline; sloped terrain needs a fitted local ground surface rather than one constant plane. When there is no ground classification or ground-plane reference, returns remain `return_unclassified`, not occupied obstacles. Rays are only useful for free-space evidence when the sensor's return/visibility semantics justify it.

### Calibrated monocular camera (optional ground coverage)

Supply a `camera` object with `fx_px`, `fy_px`, `cx_px`, `cy_px`, `image_width_px`, `image_height_px`, `camera_height_m`, and `pitch_down_deg`, plus a valid pose. This estimates where sampled image rays intersect an assumed flat ground plane. Results are explicitly labelled as assumption-based coverage; terrain relief, occlusion, camera mount offsets and calibration errors can make the estimate wrong. It does not turn an RGB detection box into a measured distance.

### GPS

For a GPS frame, provide `sensor_type: "gps"` and either top-level `latitude` / `longitude` or `gps_fix: {"latitude": ..., "longitude": ...}`. This adds a position marker; GPS positions alone do not say what obstacles occupy the cells.

## Run

```powershell
python nemotron.py --mission "Survey terrain and identify hazards"
```

Use a finer/coarser grid if appropriate:

```powershell
python nemotron.py --mission "Survey terrain" --grid-cell-size-m 0.15
```

Sensor frame input and output paths can be changed with `--sensor-dir` and `--map-dir`. The default metric resolution is 0.25 m/cell, a starting configuration only; choose based on sensor precision, rover size and intended obstacle scale.

## Safety/data semantics

- Unknown cells are not free cells.
- A 2-D LiDAR ray with a valid return contributes free-ray evidence before its endpoint and an occupied-candidate endpoint. Explicit max-range/no-return beams contribute coverage/free-ray evidence only to the reported ray extent; they do not mark an obstacle at the sensor maximum.
- A 3-D point return is not assumed to be a rock. Ground segmentation or a ground-plane reference is required to classify points as surface versus above-ground obstacle candidates; without that evidence points remain unclassified.
- The current grid is a survey/evidence map, not a complete elevation map, costmap, SLAM implementation, or verified safe route. Pose quality is recorded, but pose drift is not currently modelled probabilistically.
- Camera boxes are not projected to world coordinates from image normalized coordinates alone.
- GPS/odometry/IMU quality and coordinate-frame consistency matter. Never combine x/y poses from different origins without transforming them into the same map frame.
- No movement commands are generated by this module.
