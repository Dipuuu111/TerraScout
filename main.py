"""TerraScout application entry point.

Watches the survey-image directory and sends each *unique image content hash* to
vision_train.py once. Existing, unprocessed images are discovered on startup;
new files, replacements, and files moved into the directory are detected while
running. A file is processed only after its size and modification time remain
stable long enough to indicate that copying has finished.

Run:
    python main.py
    python main.py --input data/survey_images --database data/terrascout.db
    python main.py --once  # scan current contents and exit
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import queue
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

try:
    from watchdog.events import FileSystemEvent, FileSystemEventHandler
    from watchdog.observers import Observer
except ImportError as exc:  # pragma: no cover - depends on local installation
    raise SystemExit(
        "Missing dependency: watchdog. Install it with "
        "`python -m pip install -r requirements-vision.txt`."
    ) from exc

import vision_train as vision

LOGGER = logging.getLogger("terrascout.main")
SUPPORTED_EXTENSIONS = vision.SUPPORTED_EXTENSIONS


class SurveyIngestor:
    """Queue stable image files and ingest each globally unique SHA-256 once."""

    def __init__(
        self,
        input_dir: Path,
        database_path: Path,
        *,
        recursive: bool = True,
        stable_seconds: float = 2.0,
        file_timeout: float = 120.0,
        retry_failed: int = 2,
    ) -> None:
        self.input_dir = input_dir.expanduser().resolve()
        self.database_path = database_path.expanduser().resolve()
        self.recursive = recursive
        self.stable_seconds = stable_seconds
        self.file_timeout = file_timeout
        self.retry_failed = retry_failed

        self._queue: queue.Queue[Path | None] = queue.Queue()
        self._queued_paths: set[str] = set()
        self._queued_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._worker: threading.Thread | None = None
        self._database: vision.SurveyDatabase | None = None
        self._client: Any = None
        self._model = os.getenv("GLM_MODEL", vision.DEFAULT_MODEL).strip() or vision.DEFAULT_MODEL

    @staticmethod
    def _key(path: Path) -> str:
        # normcase makes path comparisons case-insensitive on Windows.
        return os.path.normcase(str(path.resolve()))

    def is_supported_image(self, path: Path) -> bool:
        return path.suffix.lower() in SUPPORTED_EXTENSIONS

    def enqueue(self, path: Path, *, reason: str = "filesystem event") -> None:
        """Queue one file path, coalescing duplicate create/modify notifications."""
        try:
            resolved = path.resolve()
            if not resolved.is_file() or not self.is_supported_image(resolved):
                return
            # Do not allow an event or symlink target outside the configured input tree.
            try:
                resolved.relative_to(self.input_dir)
            except ValueError:
                LOGGER.debug("Ignoring path outside survey directory: %s", resolved)
                return
        except OSError:
            # The file may have been deleted between the event and this check.
            return

        key = self._key(resolved)
        with self._queued_lock:
            if key in self._queued_paths:
                return
            self._queued_paths.add(key)
        LOGGER.debug("Queued image (%s): %s", reason, resolved)
        self._queue.put(resolved)

    def enqueue_tree(self, directory: Path) -> None:
        """Queue images under a directory, e.g. when a folder is moved into place."""
        try:
            iterator = directory.rglob("*") if self.recursive else directory.iterdir()
            for candidate in iterator:
                if candidate.is_file() and self.is_supported_image(candidate):
                    self.enqueue(candidate, reason="directory scan")
        except (FileNotFoundError, PermissionError, OSError) as exc:
            LOGGER.warning("Unable to scan directory %s: %s", directory, exc)

    def scan_existing(self) -> None:
        """Queue current contents at startup to catch files added while offline."""
        LOGGER.info("Scanning survey directory: %s", self.input_dir)
        self.enqueue_tree(self.input_dir)

    @staticmethod
    def _signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
            if not path.is_file():
                return None
            return stat.st_size, stat.st_mtime_ns
        except OSError:
            return None

    def _wait_until_stable(self, path: Path) -> tuple[int, int] | None:
        """Wait for size and mtime to stop changing before reading a copied image."""
        deadline = time.monotonic() + self.file_timeout
        previous_signature: tuple[int, int] | None = None
        unchanged_since: float | None = None

        while not self._stop_event.is_set() and time.monotonic() < deadline:
            signature = self._signature(path)
            if signature is None:
                # A short delay handles common create-before-write event ordering.
                time.sleep(0.25)
                continue
            size, _ = signature
            now = time.monotonic()
            if size <= 0:
                previous_signature = signature
                unchanged_since = now
            elif signature == previous_signature:
                if unchanged_since is not None and now - unchanged_since >= self.stable_seconds:
                    return signature
            else:
                previous_signature = signature
                unchanged_since = now
            time.sleep(0.25)

        if self._stop_event.is_set():
            return None
        if path.exists():
            LOGGER.warning("Timed out waiting for image copy to finish: %s", path.name)
        else:
            LOGGER.debug("Image disappeared before processing: %s", path.name)
        return None

    def _find_hash(self, digest: str) -> sqlite3.Row | None:
        assert self._database is not None
        return self._database.connection.execute(
            "SELECT source_path, photo_id FROM observations WHERE image_sha256 = ? LIMIT 1",
            (digest,),
        ).fetchone()

    def _find_path_row(self, source_path: str) -> sqlite3.Row | None:
        assert self._database is not None
        return self._database.connection.execute(
            "SELECT image_sha256 FROM observations WHERE source_path = ? LIMIT 1",
            (source_path,),
        ).fetchone()

    def _remove_stale_path_record(self, source_path: str, current_digest: str) -> None:
        """Remove a stale path record if its file was replaced by already-known content."""
        assert self._database is not None
        previous = self._find_path_row(source_path)
        if previous is not None and previous["image_sha256"] != current_digest:
            self._database.connection.execute(
                "DELETE FROM observations WHERE source_path = ?", (source_path,)
            )
            self._database.connection.commit()
            LOGGER.info("Removed stale observation for replaced file: %s", Path(source_path).name)

    def _process_path(self, path: Path) -> tuple[int, int] | None:
        stable_signature = self._wait_until_stable(path)
        if stable_signature is None:
            return None

        source_path = str(path.resolve())
        try:
            image_bytes = path.read_bytes()
        except OSError as exc:
            LOGGER.warning("Cannot read %s: %s", path.name, exc)
            return stable_signature
        if not image_bytes:
            LOGGER.warning("Ignoring empty image file: %s", path.name)
            return stable_signature

        digest = hashlib.sha256(image_bytes).hexdigest()
        known = self._find_hash(digest)
        if known is not None:
            self._remove_stale_path_record(source_path, digest)
            if self._key(Path(known["source_path"])) == self._key(path):
                LOGGER.info("SKIP %s (content hash already analysed)", path.name)
            else:
                LOGGER.info(
                    "SKIP %s (duplicate SHA-256; already analysed as %s)",
                    path.name,
                    known["photo_id"],
                )
            return stable_signature

        # Create the client lazily; launching main.py with an empty folder does not
        # require an API key until the first new image needs inference.
        if self._client is None:
            self._client = vision.build_client()

        last_error: Exception | None = None
        for attempt in range(self.retry_failed + 1):
            if self._stop_event.is_set():
                return stable_signature
            try:
                observation, analyzed_digest, modified_at = vision.analyze_image(
                    self._client, path, self._model
                )
                # The file can be changed while the model request is running. Record
                # the exact bytes analysed; if those bytes were already surveyed by
                # another path, don't create a duplicate observation.
                duplicate = self._find_hash(analyzed_digest)
                if duplicate is not None:
                    self._remove_stale_path_record(source_path, analyzed_digest)
                    LOGGER.info(
                        "SKIP %s (content became a duplicate during analysis; already analysed as %s)",
                        path.name,
                        duplicate["photo_id"],
                    )
                    return stable_signature

                previous = self._find_path_row(source_path)
                if previous is not None and previous["image_sha256"] != analyzed_digest:
                    # Coordinates belong to the previous image content and should not
                    # silently carry over when the file at this path is replaced.
                    self._database.connection.execute(
                        "UPDATE observations SET latitude = NULL, longitude = NULL WHERE source_path = ?",
                        (source_path,),
                    )
                    self._database.connection.commit()

                self._database.save(
                    source_path,
                    analyzed_digest,
                    modified_at,
                    self._model,
                    observation,
                )
                LOGGER.info(
                    "SAVED %s: %d observed object(s); traversal=%s; slope=%s/%s; drop-off=%s",
                    observation["photo_id"],
                    len(observation["observed_objects"]),
                    observation["terrain"]["visual_traversal_difficulty"],
                    observation["terrain"]["slope"]["assessment"],
                    observation["terrain"]["slope"]["direction"],
                    observation["terrain"]["drop_off"]["assessment"],
                )
                return stable_signature
            except Exception as exc:
                last_error = exc
                if attempt >= self.retry_failed or not self._is_transient_api_error(exc):
                    break
                delay = min(2 ** (attempt + 1), 10)
                LOGGER.warning(
                    "Temporary analysis error for %s (%s/%s): %s; retrying in %ss",
                    path.name,
                    attempt + 1,
                    self.retry_failed,
                    exc,
                    delay,
                )
                if self._stop_event.wait(delay):
                    return stable_signature

        LOGGER.error("Failed to analyse %s: %s", path.name, last_error)
        return stable_signature

    @staticmethod
    def _is_transient_api_error(exc: Exception) -> bool:
        """Retry connection/time-out/rate-limit/server errors, not malformed images."""
        transient_types = tuple(
            cls for cls in (
                getattr(vision, "APIConnectionError", None),
                getattr(vision, "APITimeoutError", None),
                getattr(vision, "RateLimitError", None),
            ) if isinstance(cls, type)
        )
        if transient_types and isinstance(exc, transient_types):
            return True
        status_error = getattr(vision, "APIStatusError", None)
        if isinstance(status_error, type) and isinstance(exc, status_error):
            status_code = getattr(exc, "status_code", None)
            return status_code == 429 or (isinstance(status_code, int) and 500 <= status_code <= 599)
        return False

    def _worker_loop(self) -> None:
        try:
            self._database = vision.SurveyDatabase(self.database_path)
            # Hash lookup is global across paths, so index it for efficient deduplication
            # as the survey grows beyond the first few photographs.
            self._database.connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_observations_image_sha256 ON observations(image_sha256)"
            )
            self._database.connection.commit()
        except Exception:
            LOGGER.exception("Could not open survey database: %s", self.database_path)
            # Drain currently queued work so queue.join() in --once mode won't hang.
            while True:
                item = self._queue.get()
                self._queue.task_done()
                if item is None:
                    return

        try:
            while True:
                path = self._queue.get()
                if path is None:
                    self._queue.task_done()
                    break
                before_signature: tuple[int, int] | None = None
                try:
                    before_signature = self._process_path(path)
                except Exception:
                    LOGGER.exception("Unexpected ingestion failure for %s", path)
                finally:
                    key = self._key(path)
                    with self._queued_lock:
                        self._queued_paths.discard(key)
                    self._queue.task_done()

                # If a writer altered the file while it was being analysed, enqueue it
                # again after removing the in-flight key. Hash dedup prevents needless calls.
                after_signature = self._signature(path)
                if before_signature is not None and after_signature is not None and after_signature != before_signature:
                    self.enqueue(path, reason="file changed during analysis")
        finally:
            if self._database is not None:
                self._database.close()
                self._database = None

    def start_worker(self) -> None:
        if self._worker is not None:
            raise RuntimeError("Image worker has already been started.")
        self._worker = threading.Thread(target=self._worker_loop, name="terrascout-vision-worker", daemon=True)
        self._worker.start()

    def wait_until_idle(self) -> None:
        self._queue.join()

    def stop(self) -> None:
        self._stop_event.set()
        if self._worker is not None:
            self._queue.put(None)
            self._worker.join()
            self._worker = None


class SurveyEventHandler(FileSystemEventHandler):
    """Translate filesystem events into image-ingestion queue entries."""

    def __init__(self, ingestor: SurveyIngestor) -> None:
        super().__init__()
        self.ingestor = ingestor

    def on_created(self, event: FileSystemEvent) -> None:
        self._handle(event, reason="created")

    def on_modified(self, event: FileSystemEvent) -> None:
        self._handle(event, reason="modified")

    def on_moved(self, event: FileSystemEvent) -> None:
        # A drag-and-drop operation may move a whole directory into the watched tree.
        destination = Path(event.dest_path) if getattr(event, "dest_path", None) else None
        if destination is None:
            return
        if getattr(event, "is_directory", False):
            self.ingestor.enqueue_tree(destination)
        else:
            self.ingestor.enqueue(destination, reason="moved into survey directory")

    def _handle(self, event: FileSystemEvent, *, reason: str) -> None:
        source = Path(event.src_path)
        if getattr(event, "is_directory", False):
            if reason == "created" and self.ingestor.recursive:
                self.ingestor.enqueue_tree(source)
            return
        self.ingestor.enqueue(source, reason=reason)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TerraScout orchestrator: automatically analyse unique survey images with GLM."
    )
    parser.add_argument(
        "--input", type=Path, default=Path("data/survey_images"),
        help="Directory to monitor for survey images (default: data/survey_images).",
    )
    parser.add_argument(
        "--database", type=Path, default=Path("data/terrascout.db"),
        help="SQLite database path (default: data/terrascout.db).",
    )
    parser.add_argument(
        "--non-recursive", action="store_true",
        help="Monitor only the top-level directory, not its subdirectories.",
    )
    parser.add_argument(
        "--once", action="store_true",
        help="Process currently present images and exit instead of watching continuously.",
    )
    parser.add_argument(
        "--stable-seconds", type=float, default=2.0,
        help="How long file size and modification time must stay unchanged (default: 2s).",
    )
    parser.add_argument(
        "--file-timeout", type=float, default=120.0,
        help="Maximum seconds to wait for a file copy to finish (default: 120s).",
    )
    parser.add_argument(
        "--retry-failed", type=int, default=2,
        help="Retries for transient API connection/time-out/rate-limit/server errors (default: 2).",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    args = parser.parse_args(argv)
    if args.stable_seconds < 0:
        parser.error("--stable-seconds cannot be negative")
    if args.file_timeout <= 0:
        parser.error("--file-timeout must be positive")
    if args.retry_failed < 0:
        parser.error("--retry-failed cannot be negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    input_dir = args.input.expanduser().resolve()
    if not input_dir.exists():
        input_dir.mkdir(parents=True, exist_ok=True)
    if not input_dir.is_dir():
        LOGGER.error("Input path is not a directory: %s", input_dir)
        return 2

    ingestor = SurveyIngestor(
        input_dir,
        args.database,
        recursive=not args.non_recursive,
        stable_seconds=args.stable_seconds,
        file_timeout=args.file_timeout,
        retry_failed=args.retry_failed,
    )
    ingestor.start_worker()

    observer: Observer | None = None
    try:
        if not args.once:
            observer = Observer()
            observer.schedule(
                SurveyEventHandler(ingestor), str(input_dir), recursive=not args.non_recursive
            )
            observer.start()
            LOGGER.info("Watching for new/changed images: %s", input_dir)
            LOGGER.info("Image SHA-256 deduplication enabled; database: %s", ingestor.database_path)

        ingestor.scan_existing()
        if args.once:
            ingestor.wait_until_idle()
            LOGGER.info("One-time scan complete.")
        else:
            while True:
                time.sleep(0.5)
    except KeyboardInterrupt:
        LOGGER.info("Shutdown requested.")
    finally:
        if observer is not None:
            observer.stop()
            observer.join(timeout=5.0)
        ingestor.stop()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
