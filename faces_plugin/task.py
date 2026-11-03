"""Background tasks: detect + embed faces, and re-identify stored faces."""
from __future__ import annotations

import csv
import io
import logging
import os
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from panoptic.core.task.task import Task

if TYPE_CHECKING:
    from .plugin import FacesPlugin

logger = logging.getLogger('Faces')

FETCH_BATCH = 32
DECODE_WORKERS = max(4, (os.cpu_count() or 8) - 2)
PREFETCH = 2
WRITE_BATCH = 200
IDENTIFY_BATCH = 2000
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff'}


def _decode(payload: tuple[str, bytes]) -> tuple[str, np.ndarray] | None:
    sha1, data = payload
    try:
        return sha1, np.asarray(Image.open(io.BytesIO(data)).convert('RGB'), dtype=np.uint8)
    except Exception as e:
        logger.warning(f'Could not decode image {sha1}: {e}')
        return None


class _BaseTask(Task):
    def _finish(self) -> None:
        self.state.running = False
        self.state.finished = True
        self._finished_event.set()
        self._notify()


class DetectFacesTask(_BaseTask):
    """Detect and embed every face of a list of images, store them, then name them."""

    def __init__(self, plugin: FacesPlugin, sha1s: list[str], recompute: bool = False):
        super().__init__()
        self.plugin = plugin
        self.project = plugin.project
        self.sha1s = sha1s
        self.recompute = recompute
        self.name = f'Faces ({plugin.params.model})'

    def start(self) -> None:
        self.state.running = True
        self._notify()

        todo = self.sha1s
        if not self.recompute:
            done = self.plugin.store.processed_sha1s()
            todo = [s for s in todo if s not in done]
        self.state.total = len(todo)
        self._notify()
        if not todo:
            self._finish()
            return

        try:
            self.set_step('Loading face models (downloaded on first use)')
            engine = self.plugin.get_engine()
        except Exception as e:
            logger.exception('Could not load the face models')
            self.set_step('Failed to load models', detail=str(e))
            self._finish()
            return

        self.state.started_at = time.monotonic()
        self.set_step('Detecting faces')
        p = self.plugin.params

        batches: Queue = Queue(maxsize=PREFETCH)
        producer = threading.Thread(target=self._producer, args=(todo, batches), daemon=True)
        producer.start()

        pending: list[tuple[str, list[dict]]] = []
        while (item := batches.get()) is not None:
            sha1s = [sha1 for sha1, _ in item]
            try:
                faces = engine.analyse_batch([rgb for _, rgb in item], p.min_detection_score,
                                             p.min_face_size)
                pending.extend(zip(sha1s, faces))
            except Exception:
                logger.exception('Face analysis failed on a batch — skipping it')
            self.state.done += len(item)
            self._notify()
            if len(pending) >= WRITE_BATCH:
                self._write(pending)
                pending = []

        self._write(pending)
        producer.join()
        self._finish()

    def _write(self, results: list[tuple[str, list[dict]]]) -> None:
        if not results:
            return
        try:
            self.plugin.store.replace_faces(results)
            self.plugin.write_identities([sha1 for sha1, _ in results])
        except Exception:
            logger.exception('Could not write a batch of face results')

    def _producer(self, sha1s: list[str], out: Queue) -> None:
        type_id = self.plugin.image_type_id
        try:
            with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
                for i in range(0, len(sha1s), FETCH_BATCH):
                    if self.is_cancelled():
                        break
                    with self.project._media_db() as db:
                        images = db.get_images(type_id=type_id, sha1=sha1s[i:i + FETCH_BATCH])
                    decoded = [r for r in pool.map(_decode, [(im.sha1, im.data) for im in images])
                               if r is not None]
                    if decoded:
                        out.put(decoded)
        finally:
            out.put(None)


class ImportGalleryTask(_BaseTask):
    """Embed a gallery folder (one sub-folder per person) as reference faces."""

    def __init__(self, plugin: FacesPlugin, root: Path):
        super().__init__()
        self.plugin = plugin
        self.root = root
        self.name = 'Import face gallery'

    def start(self) -> None:
        self.state.running = True
        self.set_step('Listing the gallery')
        photos = [(d.name, f) for d in sorted(self.root.iterdir()) if d.is_dir()
                  for f in sorted(d.iterdir()) if f.suffix.lower() in IMAGE_EXTENSIONS]
        self.state.total = len(photos)
        self._notify()

        try:
            self.set_step('Loading face models')
            engine = self.plugin.get_engine()
        except Exception as e:
            logger.exception('Could not load the face models')
            self.set_step('Failed to load models', detail=str(e))
            self._finish()
            return

        self.state.started_at = time.monotonic()
        self.set_step('Embedding reference faces')
        p = self.plugin.params
        refs: dict[str, list[tuple[str, np.ndarray]]] = defaultdict(list)
        skipped = 0
        with ThreadPoolExecutor(max_workers=DECODE_WORKERS) as pool:
            for i in range(0, len(photos), FETCH_BATCH):
                if self.is_cancelled():
                    break
                chunk = photos[i:i + FETCH_BATCH]
                decoded = list(pool.map(_read_file, [path for _, path in chunk]))
                ok = [(owner, img) for owner, img in zip(chunk, decoded) if img is not None]
                skipped += len(chunk) - len(ok)
                try:
                    found = engine.analyse_batch([img for _, img in ok], p.min_detection_score,
                                                 p.min_face_size)
                except Exception:
                    logger.exception('Face analysis failed on a gallery batch — skipping it')
                    found = [[] for _ in ok]
                for ((person, path), _), faces in zip(ok, found):
                    if not faces:
                        skipped += 1
                        continue
                    # Gallery photos are cropped around the person: the largest face is theirs.
                    face = max(faces, key=lambda f: (f['box'][2] - f['box'][0]) * (f['box'][3] - f['box'][1]))
                    refs[person].append((str(path), face['embedding']))
                self.state.done += len(chunk)
                self._notify()

        if not self.is_cancelled():
            self.plugin.store.replace_folder_refs(str(self.root), refs)
            # persons.csv (written by the gallery builder) gives each person's party.
            details = self.root / 'persons.csv'
            if details.exists():
                with details.open(encoding='utf-8') as f:
                    rows = [r for r in csv.DictReader(f) if r.get('person') in refs]
                self.plugin.store.replace_folder_persons(str(self.root), rows)
            count = sum(map(len, refs.values()))
            logger.info(f'Gallery {self.root}: {count} faces of {len(refs)} people, '
                        f'{skipped} photos without a usable face')
            self.set_step(f'Imported {count} faces of {len(refs)} people',
                          detail=f'{skipped} photos without a usable face')
            self.plugin.queue_identify()
        self._finish()


def _read_file(path: Path) -> np.ndarray | None:
    try:
        return np.asarray(Image.open(path).convert('RGB'), dtype=np.uint8)
    except Exception as e:
        logger.warning(f'Could not read {path}: {e}')
        return None


class IdentifyTask(_BaseTask):
    """Re-run gallery matching over already detected faces — no model involved."""

    def __init__(self, plugin: FacesPlugin, sha1s: list[str]):
        super().__init__()
        self.plugin = plugin
        self.sha1s = sha1s
        self.name = 'Identify faces'

    def start(self) -> None:
        self.state.running = True
        self.state.total = len(self.sha1s)
        self.set_step('Matching faces against the gallery')
        for i in range(0, len(self.sha1s), IDENTIFY_BATCH):
            if self.is_cancelled():
                break
            chunk = self.sha1s[i:i + IDENTIFY_BATCH]
            self.plugin.write_identities(chunk)
            self.state.done += len(chunk)
            self._notify()
        self._finish()
