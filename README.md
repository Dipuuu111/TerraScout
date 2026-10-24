# TerraScout

**AI-assisted terrain reconnaissance rover**

> **Project status: Paused — October 2026**

TerraScout is a self-funded rover project exploring AI-assisted terrain analysis, obstacle awareness, and mission planning. The goal is to combine computer vision, sensor measurements, and structured mission reasoning to help a rover identify terrain features and plan safer routes.

Development is temporarily paused while I focus on other projects and resolve hardware sourcing issues.

## Why development is paused

Two things have delayed further development:

- **Competing commitments:** I need to dedicate time to other ongoing projects before returning to TerraScout.
- **Hardware delivery problems:** A shipment containing key components, including the planned camera and LiDAR, was lost during shipping. This has delayed physical integration and real-world testing.

The project is paused rather than abandoned. Development can resume when time and replacement hardware are available.

## Project goals

- Analyse terrain images to identify rocks, surface characteristics, and potential obstacles.
- Distinguish terrain features from mission-specific targets instead of treating every large object as a candidate.
- Use an AI reasoning model to interpret mission objectives and prioritise observations.
- Store observations and mission information in structured local data.
- Combine camera observations with LiDAR and other sensor measurements for future spatial mapping.
- Support safer traversal by identifying potentially difficult terrain, steep slopes, and visible drop-offs, while representing uncertainty explicitly.

## Planned architecture

| Component | Intended role |
|---|---|
| Raspberry Pi 3B+ | Main onboard computer and Python orchestration |
| GLM-5.3-Flash | Vision analysis and structured terrain observations |
| NVIDIA Nemotron-3.5-Lightning | Mission reasoning and planning |
| Python and SQLite | Data processing, persistence, scoring, and mission state |
| ESP32-S3 | Sensor acquisition and communication with the main computer |
| Raspberry Pi camera | Terrain image capture |
| PONO TSD10 LiDAR | Distance measurements for obstacle awareness and future mapping |
| HC-SR04 ultrasonic sensors | Short-range obstacle experiments |

The camera and LiDAR integration remain pending. This architecture describes the intended system, not a claim that every component is currently integrated.

## Progress so far

- Implemented and tested the Python-based vision analysis pipeline using GLM-5.3-Flash.
- Developed a structured terrain-observation workflow intended to preserve features, candidate targets, and uncertainty.
- Tested NVIDIA Nemotron-based mission reasoning and generation of mission-plan data.
- Established a Python/SQLite-based approach for persistent observations and mission information.
- Successfully flashed and ran a health-check sketch on the ESP32-S3.
- Tested HC-SR04 ultrasonic sensing at 3.3 V. Initial short-range readings worked, but reliable range and operation at the intended supply voltage still need verification.

These are software and component-level milestones; autonomous rover navigation and full hardware integration have not been validated.

## Current limitations

- Camera and LiDAR integration is blocked by the lost shipment.
- The ESP32-S3's 5 V supply issue remains unresolved.
- HC-SR04 measurements have only been tested over a limited range at 3.3 V.
- Distance-based grid mapping and sensor fusion require further implementation and real-world validation.
- Terrain and mission-planning outputs need testing against real rover observations before being relied upon for navigation.

## Technology

- Python
- SQLite
- GLM-5.3-Flash
- NVIDIA Nemotron-3.5-Lightning
- Raspberry Pi
- ESP32-S3
- HC-SR04 ultrasonic sensing
- Planned CSI camera and UART LiDAR integration

## Resuming development

The next major steps are:

1. Resolve hardware sourcing and replace the missing components.
2. Complete camera and LiDAR integration.
3. Establish reliable ESP32-to-Raspberry Pi sensor communication.
4. Implement timestamped distance observations and grid-based mapping.
5. Validate terrain interpretation and mission planning using real sensor data.
6. Test safe navigation behaviour before attempting autonomous traversal.

## Project note

TerraScout is a self-funded development project. Hardware availability, budget, and time influence the pace of development. The repository records the current implementation and its limitations so work can resume without losing progress.

---

*Last status update: October 2026 — development paused.*
