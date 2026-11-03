"""Faces plugin for Panoptic — detect, identify and cluster faces with InsightFace."""

import json
import logging
import threading
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from panoptic.core.databases.data.models import (
    Property, PropertyGroup, Sha1Value, Tag, UpsertCommit,
)
from panoptic.core.databases.entity_schema import OP_CREATE
from panoptic.core.plugin.plugin import APlugin
from panoptic.models.action_models import (
    ActionContext, ActionResult, FolderPath, Group, Notif, NotifType, Score, ScoreList,
)

from .engine import DEVICES, MODELS, FaceEngine
from .matching import cluster, identify
from .store import FaceStore
from .task import DetectFacesTask, IdentifyTask, ImportGalleryTask

logger = logging.getLogger('Faces')

# key -> (property name, dtype). Order is the order shown in the UI.
PROPERTIES: dict[str, tuple[str, str]] = {
    'persons':  ('Persons',        'multi_tags'),
    'party':    ('Party',          'multi_tags'),
    'faces':    ('Faces',          'number'),
    'unknown':  ('Unknown faces',  'number'),
    'score':    ('Person score',   'number'),
}
TAG_KEYS = {'persons', 'party'}
GROUP_NAME = 'Faces'


class PluginParams(BaseModel):
    """Settings for face detection and identification.

    @compute_on_import: detect faces automatically on newly imported images
    @model: InsightFace model pack — 'buffalo_l' (best), 'buffalo_s' (fast) or 'antelopev2'
    @device: 'auto' (Core ML GPU on a Mac, CUDA if present), 'cpu', 'cuda', 'coreml' or 'ane'
    @det_size: detector input size in px — raise to 1024 for small faces in crowds
    @min_detection_score: faces detected with a lower confidence are dropped
    @min_face_size: faces smaller than this many px (shortest side) are dropped
    @match_threshold: cosine similarity needed to name a face (0.4 – 0.5 is typical)
    @match_margin: the best person must beat the second best by this much
    """
    compute_on_import: bool = False
    model: str = 'buffalo_l'
    device: str = 'auto'
    det_size: int = 640
    min_detection_score: float = 0.6
    min_face_size: int = 24
    match_threshold: float = 0.45
    match_margin: float = 0.05


class FacesPlugin(APlugin):
    """
    Face detection, identification and clustering with InsightFace (RetinaFace + ArcFace).

    Faces are stored by the plugin itself (several per image, each with its box) and
    surfaced in Panoptic as properties: 'Persons' and 'Party' (tags), 'Faces',
    'Unknown faces'. The party comes from the gallery's persons.csv.

    Workflow: run **detect_faces**, build a gallery with **add_reference** (selected
    images) or **import_gallery** (one sub-folder per person), and every stored face is
    re-matched. **cluster_faces** groups unknown people without any gallery, and
    **find_similar_faces** searches by face rather than by whole image.

    The default 'buffalo_l' weights are for non-commercial research use only.
    """

    def __init__(self, name: str, project, plugin_path: str):
        super().__init__(name=name, project=project, plugin_path=plugin_path)
        self.params = PluginParams()
        # faces.db holds face embeddings (biometric data): keep it out of any git repository
        # the project folder may end up in.
        _ignore_folder(self.data_path)
        self.store = FaceStore(self.data_path / 'faces.db')

        self._engine: FaceEngine | None = None
        self._engine_key: tuple | None = None
        self._engine_lock = threading.Lock()
        # Reentrant: write_identities holds it and calls _ensure_properties, which takes it too.
        self._write_lock = threading.RLock()
        self._props: dict[str, Property] | None = None
        self.image_type_id: int = 0

        self.project.on_import_complete(self._on_import)
        self.add_action_easy(self.detect_faces, ['execute'])
        self.add_action_easy(self.add_reference, ['execute'])
        self.add_action_easy(self.import_gallery, ['execute'])
        self.add_action_easy(self.remove_person, ['execute'])
        self.add_action_easy(self.forget_faces, ['execute'])
        self.add_action_easy(self.identify_faces, ['execute'])
        self.add_action_easy(self.cluster_faces, ['group'])
        self.add_action_easy(self.find_similar_faces, ['similar'])

    def _start(self) -> None:
        self.image_type_id = self._largest_image_type()

    def _stop(self) -> None:
        if self._engine is not None:
            self._engine.close()
        self._engine = None
        self.store.close()

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def detect_faces(self, context: ActionContext, recompute: bool = False) -> ActionResult:
        """Detect the faces of the selected images, embed them and name the known ones.
        @recompute: also re-process images that were already analysed
        """
        if error := self._validate_params():
            return _error('BadParams', error)
        sha1s = self._sha1s(context)
        if not sha1s:
            return _error('NoData', 'No images found')
        self._ensure_properties()
        task = self.project.add_task(DetectFacesTask(self, sha1s, recompute=recompute))
        return ActionResult(task_id=task.id, notifs=[_info(f'Started face detection on {len(sha1s)} images')])

    def add_reference(self, context: ActionContext, person: str = '') -> ActionResult:
        """Use the selected images as reference faces of a person, then re-identify.

        Each image contributes its largest face, so pick portraits or shots where the
        person is clearly the main subject. Run detect_faces on them first.
        @person: name of the person, as it should appear in the Persons tags
        """
        person = (person or '').strip()
        if not person:
            return _error('BadParams', 'Give the name of the person')
        sha1s = self._sha1s(context)
        faces, emb = self.store.faces(sha1s)
        largest: dict[str, int] = {}
        for i, f in enumerate(faces):
            if f.sha1 not in largest or _area(f.box) > _area(faces[largest[f.sha1]].box):
                largest[f.sha1] = i
        if not largest:
            return _error('NoFaces', 'No detected faces in the selection — run detect_faces first')

        self.store.add_refs(person, [(sha1, emb[i]) for sha1, i in largest.items()])
        return self._reidentify(f'Added {len(largest)} reference faces for {person}')

    def import_gallery(self, context: ActionContext, folder: FolderPath = None) -> ActionResult:
        """Import reference faces from a folder with one sub-folder per person.

        The sub-folder name is the person name, e.g. gallery/Emmanuel Macron/*.jpg. The
        largest face of each photo is used; photos without a face are skipped. Importing
        the same folder again replaces what it imported before.
        @folder: root folder of the gallery
        """
        if error := self._validate_params():
            return _error('BadParams', error)
        if not folder:
            return _error('BadParams', 'Choose the gallery folder (one sub-folder per person)')
        root = Path(folder).expanduser().resolve()
        if not root.is_dir():
            return _error('BadParams', f'{root} is not a folder')
        self._ensure_properties()
        task = self.project.add_task(ImportGalleryTask(self, root))
        return ActionResult(task_id=task.id, notifs=[_info(f'Importing the gallery {root}')])

    def remove_person(self, context: ActionContext, person: str = '') -> ActionResult:
        """Remove every reference face of a person from the gallery, then re-identify.
        @person: name of the person to remove
        """
        person = (person or '').strip()
        if not person:
            return _error('BadParams', 'Give the name of the person')
        removed = self.store.remove_person(person)
        if not removed:
            return _error('NotFound', f'No reference faces for {person!r}')
        return self._reidentify(f'Removed {removed} reference faces of {person}')

    def forget_faces(self, context: ActionContext, everything: bool = False) -> ActionResult:
        """Erase the detected faces of the selected images and clear their Faces properties.

        For a request to erase someone's data (GDPR Art. 17), or when the analysis is no
        longer needed. The gallery is kept: remove_person erases a person from it.
        @everything: erase every stored face of the project, selected or not
        """
        if everything:
            sha1s = list({i.sha1 for i in self.project.get_instances() if i.sha1})
            removed = self.store.delete_faces(None)
        else:
            if not context.instance_ids:
                return _error('NoData', 'Select the images whose faces should be erased, '
                                        'or tick "everything"')
            sha1s = self._sha1s(context)
            removed = self.store.delete_faces(sha1s)
        empty = {'persons': [], 'party': [], 'faces': None, 'unknown': None, 'score': None}
        for i in range(0, len(sha1s), 2000):
            self._write([(s, empty) for s in sha1s[i:i + 2000]])
        return ActionResult(notifs=[_info(f'Erased {removed} faces from {len(sha1s)} images')])

    def identify_faces(self, context: ActionContext) -> ActionResult:
        """Re-match the stored faces against the gallery, e.g. after changing the threshold."""
        return self._reidentify('Re-identifying faces')

    def cluster_faces(self, context: ActionContext, min_similarity: float = 0.5,
                      min_size: int = 3, unknown_only: bool = True) -> ActionResult:
        """Group the selected images by the people in them, without any gallery.

        An image with several people appears in several groups. Name a group, then use
        add_reference on it to grow the gallery.
        @min_similarity: cosine similarity for a face to join a cluster (0.45 – 0.6)
        @min_size: smallest cluster to keep, in faces
        @unknown_only: only cluster faces the gallery could not name
        """
        faces, emb = self.store.faces(self._sha1s(context))
        if unknown_only and len(faces):
            names, refs = self.store.refs()
            p = self.params
            known = [n is not None for n, _ in identify(emb, names, refs,
                                                        p.match_threshold, p.match_margin)]
            keep = [i for i, k in enumerate(known) if not k]
            faces, emb = [faces[i] for i in keep], emb[keep]
        if not faces:
            return _error('NoFaces', 'No faces to cluster — run detect_faces first')

        groups = []
        for members in cluster(emb, min_similarity):
            if len(members) < min_size:
                break
            # Cohesion: mean similarity of the members to their centroid.
            centroid = emb[members].mean(axis=0)
            centroid /= np.linalg.norm(centroid)
            cohesion = float((emb[members] @ centroid).mean())
            sha1s = list(dict.fromkeys(faces[i].sha1 for i in members))
            groups.append(Group(sha1s=sha1s, name=f'Person {len(groups) + 1}',
                                score=Score(value=round(cohesion, 3), min=0, max=1,
                                            description='cohesion')))
        if not groups:
            return _error('NoClusters', f'No cluster with at least {min_size} faces')
        return ActionResult(groups=groups)

    def find_similar_faces(self, context: ActionContext, max_results: int = 200) -> ActionResult:
        """Find images containing a face similar to one in the selected images.
        @max_results: maximum number of images to return
        """
        selected = self._sha1s(context)
        _, query = self.store.faces(selected)
        if not len(query):
            return _error('NoFaces', 'No detected faces in the selection')
        faces, emb = self.store.faces()

        best: dict[str, float] = defaultdict(lambda: -1.0)
        sims = (emb @ query.T).max(axis=1)
        for f, s in zip(faces, sims):
            best[f.sha1] = max(best[f.sha1], float(s))
        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[:max_results]
        return ActionResult(groups=[Group(
            sha1s=[s for s, _ in ranked],
            scores=ScoreList(values=[round(v, 4) for _, v in ranked], min=0, max=1,
                             description='face similarity'),
        )])

    # ------------------------------------------------------------------

    def _on_import(self, root_folder_id: int | None = None) -> None:
        if not self.params.compute_on_import or self._validate_params():
            return
        sha1s = list({i.sha1 for i in self.project.get_instances() if i.sha1})
        if sha1s:
            self._ensure_properties()
            self.project.add_task(DetectFacesTask(self, sha1s))

    def _reidentify(self, message: str) -> ActionResult:
        self._ensure_properties()
        processed = sorted(self.store.processed_sha1s())
        counts = self.store.ref_counts()
        summary = (f'{len(counts)} people, {sum(counts.values())} faces' if len(counts) > 10 else
                   ', '.join(f'{p} ({n})' for p, n in counts.items()) or 'empty')
        if not processed:
            # Nothing detected yet: the gallery is kept, and detect_faces names the faces itself.
            return ActionResult(notifs=[_info(f'{message}. Gallery: {summary}. No images analysed '
                                              f'yet — run detect_faces to find and name faces')])
        task = self.queue_identify()
        return ActionResult(task_id=task.id, notifs=[_info(f'{message}. Gallery: {summary}')])

    def queue_identify(self):
        """Queue a re-match of every analysed image against the current gallery."""
        return self.project.add_task(IdentifyTask(self, sorted(self.store.processed_sha1s())))

    def _validate_params(self) -> str | None:
        p = self.params
        if p.model not in MODELS:
            return f'Unknown model {p.model!r}, expected one of {list(MODELS)}'
        if p.device not in DEVICES:
            return f'Unknown device {p.device!r}, expected one of {list(DEVICES)}'
        if p.det_size < 160 or p.det_size % 32:
            return 'det_size must be at least 160 and a multiple of 32'
        if not -1 <= p.match_threshold <= 1:
            return 'match_threshold must be between -1 and 1'
        return None

    def _sha1s(self, context: ActionContext) -> list[str]:
        instances = (self.project.get_instances(id=context.instance_ids)
                     if context.instance_ids else self.project.get_instances())
        return list({i.sha1 for i in instances if i.sha1})

    # ------------------------------------------------------------------
    # Model engine (rebuilt when its settings change)
    # ------------------------------------------------------------------

    def get_engine(self) -> FaceEngine:
        p = self.params
        key = (p.model, p.device, p.det_size)
        with self._engine_lock:
            if self._engine is None or self._engine_key != key:
                if self._engine is not None:
                    self._engine.close()
                self._engine = FaceEngine(self.data_path / 'insightface', p.model,
                                          p.device, p.det_size)
                self._engine_key = key
            return self._engine

    # ------------------------------------------------------------------
    # Writing results
    # ------------------------------------------------------------------

    def write_identities(self, sha1s: list[str]) -> None:
        """Match the stored faces of ``sha1s`` against the gallery and write the properties."""
        if not sha1s:
            return
        p = self.params
        faces, emb = self.store.faces(sha1s)
        names, refs = self.store.refs()
        matches = identify(emb, names, refs, p.match_threshold, p.match_margin)
        parties = self.store.parties()

        results: dict[str, dict] = {s: {'persons': [], 'party': [], 'faces': 0, 'unknown': 0,
                                        'score': None} for s in sha1s}
        for face, (person, score) in zip(faces, matches):
            res = results[face.sha1]
            res['faces'] += 1
            if person is None:
                res['unknown'] += 1
                continue
            if person not in res['persons']:
                res['persons'].append(person)
            party = parties.get(person)
            if party and party not in res['party']:
                res['party'].append(party)
            # The weakest naming of the image: a low value flags images worth checking.
            res['score'] = round(min(res['score'] or 1.0, score), 3)
        self._write(list(results.items()))

    def _write(self, results: list[tuple[str, dict]]) -> None:
        props = self._ensure_properties()
        with self._write_lock:
            tag_ids = {key: self._ensure_tags(props[key], [n for _, r in results for n in r[key]])
                       for key in TAG_KEYS}
            commit = UpsertCommit()
            for sha1, res in results:
                for key, prop in props.items():
                    value = res.get(key)
                    if key in TAG_KEYS:
                        value = [tag_ids[key][n] for n in value]
                    commit.sha1_values.setdefault(prop.id, []).append(
                        Sha1Value(property_id=prop.id, sha1=sha1, value=value,
                                  commit_id=0, operation=OP_CREATE))
            self.project.apply_upsert_commit(commit)

    # ------------------------------------------------------------------
    # Properties and tags
    # ------------------------------------------------------------------

    @property
    def _prop_cache_file(self) -> Path:
        return self.data_path / 'properties.json'

    def _ensure_properties(self) -> dict[str, Property]:
        """Return the plugin's properties, creating any that are missing.

        Ids are cached on disk: nothing in the data model records which plugin created
        a property, so the cache is how we find ours again after a restart.
        """
        if self._props and all(k in self._props for k in PROPERTIES):
            return self._props
        with self._write_lock:
            cache: dict = {}
            if self._prop_cache_file.exists():
                try:
                    cache = json.loads(self._prop_cache_file.read_text())
                except Exception:
                    cache = {}
            cached_ids = cache.get('properties', {})
            group_id = cache.get('group_id')

            existing = {p.id: p for p in self.project.get_properties()}
            props: dict[str, Property] = {}
            for key, (name, dtype) in PROPERTIES.items():
                prop = existing.get(cached_ids.get(key, -1))
                if prop is None or prop.dtype != dtype:
                    group_id = self._ensure_property_group(group_id)
                    prop = self.project.add_property(name=name, dtype=dtype, mode='sha1',
                                                     property_group_id=group_id)
                props[key] = prop

            self._props = props
            self._prop_cache_file.write_text(json.dumps({
                'group_id': group_id,
                'properties': {k: p.id for k, p in props.items()},
            }))
            return props

    def _ensure_property_group(self, cached_id: int | None) -> int | None:
        """Create (once) the property group. Best-effort: failing only costs the grouping."""
        if cached_id is not None:
            return cached_id
        try:
            with self.project._project_db() as pdb:
                gid = pdb.allocate_property_groups(1)
            gid = gid if isinstance(gid, int) else list(gid)[0]
            commit = UpsertCommit()
            commit.property_groups[gid] = PropertyGroup(id=gid, name=GROUP_NAME,
                                                        commit_id=0, operation=OP_CREATE)
            self.project.apply_upsert_commit(commit)
            return gid
        except Exception:
            logger.warning('Could not create the Faces property group', exc_info=True)
            return None

    def _ensure_tags(self, prop: Property, values: list[str]) -> dict[str, int]:
        """Map tag labels to ids for ``prop``, creating the missing ones in one commit."""
        existing = {t.value: t.id for t in self.project.get_tags(list_id=prop.tag_list_id)}
        missing = [v for v in dict.fromkeys(values) if v not in existing]
        if not missing:
            return existing

        with self.project._project_db() as pdb:
            allocated = pdb.allocate_tags(len(missing))
        ids = [allocated] if isinstance(allocated, int) else list(allocated)

        commit = UpsertCommit()
        for tag_id, value in zip(ids, missing):
            # A stable color in the UI's 0..11 range, so a person keeps theirs across projects.
            commit.tags[tag_id] = Tag(id=tag_id, list_id=prop.tag_list_id, parents=[],
                                      value=value, color=zlib.crc32(value.encode()) % 12,
                                      commit_id=0, operation=OP_CREATE)
            existing[value] = tag_id
        self.project.apply_upsert_commit(commit)
        return existing

    def _largest_image_type(self) -> int:
        """Id of the biggest stored image variant — small faces need every pixel."""
        with self.project._media_db() as db:
            types = db.get_image_types()
        if not types:
            return 0
        return max(types, key=lambda t: (t.width or 0) * (t.height or 0)).id


def _ignore_folder(folder: Path) -> None:
    """Write a .gitignore that ignores everything in ``folder`` (best effort)."""
    try:
        marker = folder / '.gitignore'
        if not marker.exists():
            marker.write_text('# Personal data (faces, biometric embeddings): never publish\n*\n')
    except OSError:
        logger.warning(f'Could not write {folder}/.gitignore', exc_info=True)


def _area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _info(message: str) -> Notif:
    return Notif(NotifType.INFO, name='Faces', message=message)


def _error(name: str, message: str) -> ActionResult:
    return ActionResult(notifs=[Notif(NotifType.ERROR, name=name, message=message)])
