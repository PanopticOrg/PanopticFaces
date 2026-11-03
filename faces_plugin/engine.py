"""InsightFace pipeline: RetinaFace (SCRFD) detection + ArcFace embedding.

InsightFace's own model classes do the pre- and post-processing (letterbox, NMS,
alignment); this module only chooses where the networks run and batches ArcFace over
every face of a batch of images, instead of one call per face.

On Apple Silicon both networks run through Core ML on the GPU. Core ML wants static
shapes, so fixed-shape copies of the ONNX files are written next to the originals (the
detector at det_size, ArcFace at REC_BATCH). M4 Pro, one image of 640 px:

    detector   CPU 37 ms     Core ML GPU  7 ms     Neural Engine 62 ms
    ArcFace    CPU 17 ms/face (batch 1)   Core ML GPU 1.9 ms/face (batch 32)

The Neural Engine is slower than the CPU on these convolutional networks, so 'auto'
does not use it ('ane' does, for testing). Stability: Core ML is used only once its
outputs match the CPU's on a probe; if it fails later, the engine falls back to the CPU
for the rest of the session.

The 'buffalo_l' weights are licensed for non-commercial research use only.
"""
from __future__ import annotations

import glob
import logging
import os
import platform
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

logger = logging.getLogger('Faces')

MODELS = ('buffalo_l', 'buffalo_s', 'antelopev2')
DEVICES = ('auto', 'cpu', 'cuda', 'coreml', 'ane')
REC_BATCH = 32               # ArcFace batch on Core ML; the last one is padded
DET_WORKERS = 3              # detector calls in flight: overlaps NMS on the CPU with the GPU
MIN_PROBE_COSINE = 0.99
COREML_UNITS = {'coreml': 'CPUAndGPU', 'ane': 'CPUAndNeuralEngine'}


def _apple_silicon() -> bool:
    return sys.platform == 'darwin' and platform.machine() == 'arm64'


def resolve_device(device: str) -> str:
    """The concrete backend for a device setting: 'cpu', 'cuda', 'coreml' or 'ane'."""
    import onnxruntime
    available = onnxruntime.get_available_providers()
    if device == 'auto':
        if 'CUDAExecutionProvider' in available:
            return 'cuda'
        if _apple_silicon() and 'CoreMLExecutionProvider' in available:
            return 'coreml'
        return 'cpu'
    if device in COREML_UNITS and 'CoreMLExecutionProvider' not in available:
        logger.warning('Core ML is not available in this onnxruntime, using the CPU')
        return 'cpu'
    if device == 'cuda' and 'CUDAExecutionProvider' not in available:
        logger.warning('CUDA is not available in this onnxruntime, using the CPU')
        return 'cpu'
    return device


def _fixed_copy(src: str, shape: tuple[int, ...], out_dir: Path) -> str:
    """A copy of ``src`` with a static input shape, as Core ML needs (written once)."""
    import onnx
    from onnxruntime.tools.onnx_model_utils import make_input_shape_fixed

    path = out_dir / f'{Path(src).stem}_{"x".join(map(str, shape))}.onnx'
    if not path.exists():
        out_dir.mkdir(parents=True, exist_ok=True)
        model = onnx.load(src)
        make_input_shape_fixed(model.graph, model.graph.input[0].name, list(shape))
        # The declared outputs say batch 1 whatever the input; make them agree.
        for output in model.graph.output:
            dim = output.type.tensor_type.shape.dim
            if dim and dim[0].dim_value == 1 and shape[0] != 1:
                dim[0].dim_value = shape[0]
        tmp = path.with_suffix('.tmp')
        onnx.save(model, str(tmp))
        os.replace(tmp, path)
    return str(path)


class FaceEngine:
    def __init__(self, root: Path, model: str, device: str, det_size: int):
        from insightface.model_zoo import get_model
        from insightface.utils.storage import ensure_available

        model_dir = ensure_available('models', model, root=str(root))
        # Find the detector and the recogniser of the pack by their task, as FaceAnalysis does.
        self.files: dict[str, str] = {}
        for path in sorted(glob.glob(os.path.join(model_dir, '*.onnx'))):
            m = get_model(path, providers=['CPUExecutionProvider'])
            if m is not None and m.taskname in ('detection', 'recognition'):
                self.files.setdefault(m.taskname, path)
        if set(self.files) != {'detection', 'recognition'}:
            raise RuntimeError(f'{model}: no detector or recogniser in {model_dir}')

        self.det_size = det_size
        self.fixed_dir = Path(model_dir) / 'fixed'
        self.backend = resolve_device(device)
        self._pool = ThreadPoolExecutor(max_workers=DET_WORKERS, thread_name_prefix='faces-det')
        self._fallback_lock = threading.Lock()

        if self.backend in COREML_UNITS:
            try:
                self._build(self.backend)
                self._check_against_cpu()
            except Exception as e:
                logger.warning(f'Core ML is unusable for {model} ({e}), using the CPU')
                self.backend = 'cpu'
        if self.backend not in COREML_UNITS:
            self._build(self.backend)
        logger.info(f'InsightFace {model} on {self.backend}')

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def _session(self, path: str, backend: str):
        import onnxruntime
        opts = onnxruntime.SessionOptions()
        opts.log_severity_level = 3
        if backend in COREML_UNITS:
            cache = self.fixed_dir / f'coreml_{COREML_UNITS[backend]}'
            cache.mkdir(parents=True, exist_ok=True)
            providers = [('CoreMLExecutionProvider', {
                'ModelFormat': 'MLProgram',
                'MLComputeUnits': COREML_UNITS[backend],
                'RequireStaticInputShapes': '1',
                'ModelCacheDirectory': str(cache),
            }), 'CPUExecutionProvider']
        elif backend == 'cuda':
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        return onnxruntime.InferenceSession(path, sess_options=opts, providers=providers)

    def _build(self, backend: str) -> None:
        from insightface.model_zoo.arcface_onnx import ArcFaceONNX
        from insightface.model_zoo.scrfd import SCRFD

        det_file, rec_file = self.files['detection'], self.files['recognition']
        if backend in COREML_UNITS:
            det_file = _fixed_copy(det_file, (1, 3, self.det_size, self.det_size), self.fixed_dir)
            rec_file = _fixed_copy(rec_file, (REC_BATCH, 3, 112, 112), self.fixed_dir)
            self.rec_batch = REC_BATCH
        else:
            self.rec_batch = 0                       # dynamic: one call for all the faces

        det = SCRFD(model_file=det_file, session=self._session(det_file, backend))
        det.prepare(0, input_size=(self.det_size, self.det_size), det_thresh=0.5)
        rec = ArcFaceONNX(model_file=rec_file, session=self._session(rec_file, backend))
        self.det, self.rec = det, rec

    def _check_against_cpu(self) -> None:
        """Compare Core ML with the CPU on a probe; raise when they disagree."""
        rng = np.random.default_rng(1234)
        size = self.det_size
        det_in = rng.normal(0, 1, (1, 3, size, size)).astype(np.float32)
        rec_in = rng.normal(0, 1, (REC_BATCH, 3, 112, 112)).astype(np.float32)

        cpu_det = self._session(self.files['detection'], 'cpu')
        cpu_rec = self._session(self.files['recognition'], 'cpu')
        name = self.det.input_name
        ref = cpu_det.run(None, {name: det_in})
        got = self.det.session.run(None, {name: det_in})
        det_diff = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))

        name = self.rec.input_name
        ref = cpu_rec.run(None, {name: rec_in})[0]
        got = self.rec.session.run(None, {name: rec_in})[0]
        if not (np.isfinite(got).all() and np.isfinite(det_diff)):
            raise FloatingPointError('non-finite outputs on the probe')
        ref = ref / np.linalg.norm(ref, axis=1, keepdims=True)
        got = got / np.linalg.norm(got, axis=1, keepdims=True)
        cosine = float((ref * got).sum(axis=1).min())
        logger.info(f'Core ML probe: ArcFace cosine {cosine:.4f}, detector max diff {det_diff:.4f}')
        if cosine < MIN_PROBE_COSINE or det_diff > 0.05:
            raise ValueError(f'outputs differ from the CPU (cosine {cosine:.3f}, '
                             f'detector diff {det_diff:.3f})')

    def _fall_back(self, error: Exception) -> None:
        with self._fallback_lock:
            if self.backend in COREML_UNITS:
                logger.warning(f'Core ML failed ({error}), using the CPU from now on')
                self.backend = 'cpu'
                self._build('cpu')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    def analyse(self, rgb: np.ndarray, min_score: float, min_size: int) -> list[dict]:
        return self.analyse_batch([rgb], min_score, min_size)[0]

    def analyse_batch(self, images: list[np.ndarray], min_score: float,
                      min_size: int) -> list[list[dict]]:
        """Faces of each RGB image: relative box, detector score, L2-normalised embedding."""
        try:
            return self._analyse_batch(images, min_score, min_size)
        except Exception as e:
            if self.backend not in COREML_UNITS:
                raise
            self._fall_back(e)
            return self._analyse_batch(images, min_score, min_size)

    def _analyse_batch(self, images, min_score, min_size):
        from insightface.utils import face_align

        bgrs = [np.ascontiguousarray(rgb[:, :, ::-1]) for rgb in images]   # InsightFace wants BGR
        detections = list(self._pool.map(lambda im: self.det.detect(im, max_num=0), bgrs))

        results: list[list[dict]] = [[] for _ in images]
        crops: list[np.ndarray] = []
        owners: list[dict] = []
        for bgr, (bboxes, kpss), faces in zip(bgrs, detections, results):
            h, w = bgr.shape[:2]
            for i in range(bboxes.shape[0]):
                x1, y1, x2, y2, score = (float(v) for v in bboxes[i])
                if score < min_score or min(x2 - x1, y2 - y1) < min_size:
                    continue
                face = {'box': (max(x1, 0) / w, max(y1, 0) / h, min(x2, w) / w, min(y2, h) / h),
                        'score': score}
                faces.append(face)
                crops.append(face_align.norm_crop(bgr, landmark=kpss[i], image_size=112))
                owners.append(face)

        for face, emb in zip(owners, self._embed(crops)):
            face['embedding'] = emb
        return results

    def _embed(self, crops: list[np.ndarray]) -> np.ndarray:
        if not crops:
            return np.zeros((0, 512), dtype=np.float32)
        step = self.rec_batch or len(crops)
        out = []
        for i in range(0, len(crops), step):
            chunk = crops[i:i + step]
            n = len(chunk)
            if self.rec_batch and n < step:               # the fixed-shape model needs a full batch
                chunk = chunk + [np.zeros_like(chunk[0])] * (step - n)
            out.append(self.rec.get_feat(chunk)[:n])
        emb = np.concatenate(out).astype(np.float32)
        if not np.isfinite(emb).all():
            raise FloatingPointError('ArcFace returned non-finite values')
        return emb / np.linalg.norm(emb, axis=1, keepdims=True)
