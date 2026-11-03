"""Plugin-owned face store: one SQLite file in the plugin data folder.

Panoptic keeps one vector per image and vector type, so faces (0..N per image, each
with a box) live here instead. The table shape — sha1, box, score, embedding — is the
one a future core "region" entity would need, so moving to it later is a storage swap.
"""
from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

EMBED_DIM = 512


@dataclass
class Face:
    id: int
    sha1: str
    box: tuple[float, float, float, float]   # x1, y1, x2, y2, relative to image size (0..1)
    score: float                             # detector confidence


def _to_blob(v: np.ndarray) -> bytes:
    return np.asarray(v, dtype=np.float32).tobytes()


def _from_blobs(blobs: list[bytes]) -> np.ndarray:
    if not blobs:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)
    return np.frombuffer(b''.join(blobs), dtype=np.float32).reshape(len(blobs), -1)


class FaceStore:
    def __init__(self, path: Path):
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.executescript('''
            PRAGMA journal_mode = WAL;
            -- deleted faces are overwritten with zeros, not just unlinked (GDPR erasure)
            PRAGMA secure_delete = ON;
            CREATE TABLE IF NOT EXISTS faces (
                id    INTEGER PRIMARY KEY,
                sha1  TEXT NOT NULL,
                x1 REAL, y1 REAL, x2 REAL, y2 REAL,
                score REAL,
                embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS faces_sha1 ON faces(sha1);
            -- every analysed image, including those without any face
            CREATE TABLE IF NOT EXISTS processed (sha1 TEXT PRIMARY KEY);
            -- the reference gallery: known faces of named persons
            CREATE TABLE IF NOT EXISTS refs (
                id     INTEGER PRIMARY KEY,
                person TEXT NOT NULL,
                source TEXT,              -- sha1 or file path the face came from
                embedding BLOB NOT NULL
            );
            CREATE INDEX IF NOT EXISTS refs_person ON refs(person);
            -- what a gallery's persons.csv says about each person (party, group…)
            CREATE TABLE IF NOT EXISTS persons (
                person TEXT PRIMARY KEY,
                root   TEXT,               -- gallery folder it was imported from
                qid    TEXT,
                party  TEXT,
                grp    TEXT
            );
        ''')

    def close(self) -> None:
        with self._lock:
            self._db.close()

    # ------------------------------------------------------------------
    # Faces
    # ------------------------------------------------------------------

    def replace_faces(self, results: list[tuple[str, list[dict]]]) -> None:
        """Store the faces of a batch of images, replacing what these images had."""
        with self._lock, self._db:
            for sha1, faces in results:
                self._db.execute('DELETE FROM faces WHERE sha1 = ?', (sha1,))
                self._db.executemany(
                    'INSERT INTO faces (sha1, x1, y1, x2, y2, score, embedding) '
                    'VALUES (?, ?, ?, ?, ?, ?, ?)',
                    [(sha1, *f['box'], f['score'], _to_blob(f['embedding'])) for f in faces],
                )
                self._db.execute('INSERT OR IGNORE INTO processed (sha1) VALUES (?)', (sha1,))

    def delete_faces(self, sha1s: list[str] | None = None) -> int:
        """Erase the faces of ``sha1s`` (every face when None) and forget the images were
        analysed. Returns the number of faces deleted."""
        with self._lock:
            with self._db:
                if sha1s is None:
                    removed = self._db.execute('DELETE FROM faces').rowcount
                    self._db.execute('DELETE FROM processed')
                else:
                    removed = 0
                    for i in range(0, len(sha1s), 900):    # SQLite variable limit
                        chunk = sha1s[i:i + 900]
                        marks = ','.join('?' * len(chunk))
                        removed += self._db.execute(f'DELETE FROM faces WHERE sha1 IN ({marks})',
                                                    chunk).rowcount
                        self._db.execute(f'DELETE FROM processed WHERE sha1 IN ({marks})', chunk)
            self._checkpoint()
        return removed

    def _checkpoint(self) -> None:
        """Flush and empty the WAL file, so erased data does not linger in it."""
        self._db.execute('PRAGMA wal_checkpoint(TRUNCATE)')

    def processed_sha1s(self) -> set[str]:
        with self._lock:
            return {r[0] for r in self._db.execute('SELECT sha1 FROM processed')}

    def faces(self, sha1s: list[str] | None = None) -> tuple[list[Face], np.ndarray]:
        """Faces of ``sha1s`` (all when None) and their embeddings, row-aligned."""
        query = 'SELECT id, sha1, x1, y1, x2, y2, score, embedding FROM faces'
        with self._lock:
            if sha1s is None:
                rows = self._db.execute(query + ' ORDER BY id').fetchall()
            else:
                rows = []
                for i in range(0, len(sha1s), 900):    # SQLite variable limit
                    chunk = sha1s[i:i + 900]
                    marks = ','.join('?' * len(chunk))
                    rows += self._db.execute(f'{query} WHERE sha1 IN ({marks}) ORDER BY id',
                                             chunk).fetchall()
        faces = [Face(r[0], r[1], (r[2], r[3], r[4], r[5]), r[6]) for r in rows]
        return faces, _from_blobs([r[7] for r in rows])

    # ------------------------------------------------------------------
    # Gallery
    # ------------------------------------------------------------------

    def add_refs(self, person: str, items: list[tuple[str, np.ndarray]]) -> None:
        """Add ``(source, embedding)`` reference faces for ``person``."""
        with self._lock, self._db:
            self._db.executemany(
                'INSERT INTO refs (person, source, embedding) VALUES (?, ?, ?)',
                [(person, source, _to_blob(e)) for source, e in items],
            )

    def replace_folder_refs(self, root: str, refs: dict[str, list[tuple[str, np.ndarray]]]) -> None:
        """Replace every reference imported from the folder ``root`` with ``refs``
        (person -> [(file path, embedding)]). References added from images are kept."""
        prefix = root.rstrip('/') + '/'
        with self._lock, self._db:
            self._db.execute('DELETE FROM refs WHERE substr(source, 1, ?) = ?', (len(prefix), prefix))
            self._db.executemany(
                'INSERT INTO refs (person, source, embedding) VALUES (?, ?, ?)',
                [(person, source, _to_blob(e)) for person, items in refs.items() for source, e in items],
            )

    def replace_folder_persons(self, root: str, rows: list[dict]) -> None:
        """Replace the person details imported from the gallery folder ``root``."""
        with self._lock, self._db:
            self._db.execute('DELETE FROM persons WHERE root = ?', (root,))
            self._db.executemany(
                'INSERT OR REPLACE INTO persons (person, root, qid, party, grp) VALUES (?, ?, ?, ?, ?)',
                [(r['person'], root, r.get('qid') or None, r.get('party') or None,
                  r.get('group') or None) for r in rows],
            )

    def parties(self) -> dict[str, str]:
        with self._lock:
            return dict(self._db.execute('SELECT person, party FROM persons WHERE party IS NOT NULL'))

    def remove_person(self, person: str) -> int:
        with self._lock:
            with self._db:
                removed = self._db.execute('DELETE FROM refs WHERE person = ?', (person,)).rowcount
                self._db.execute('DELETE FROM persons WHERE person = ?', (person,))
            self._checkpoint()
        return removed

    def refs(self) -> tuple[list[str], np.ndarray]:
        """Person name of every reference face, and the embeddings, row-aligned."""
        with self._lock:
            rows = self._db.execute('SELECT person, embedding FROM refs ORDER BY id').fetchall()
        return [r[0] for r in rows], _from_blobs([r[1] for r in rows])

    def ref_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._db.execute(
                'SELECT person, COUNT(*) FROM refs GROUP BY person ORDER BY person'))
