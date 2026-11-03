"""Pure numpy face matching and clustering. Embeddings are L2-normalised, so a dot
product is the cosine similarity."""
from __future__ import annotations

import numpy as np

CHUNK = 1024                 # faces compared at once; bounds memory with a large gallery


def identify(embeddings: np.ndarray, ref_names: list[str], refs: np.ndarray,
             threshold: float, margin: float) -> list[tuple[str | None, float]]:
    """Name each face after its nearest reference person, or None when unsure.

    A face is named when its best person scores at least ``threshold`` and beats the
    second-best person by ``margin`` — look-alikes are left unknown rather than guessed.
    """
    if not len(embeddings):
        return []
    if not len(refs):
        return [(None, 0.0)] * len(embeddings)

    # Group the references by person, so the best reference of each person is one
    # np.maximum.reduceat over contiguous columns (nearest neighbour per person, more
    # robust than a centroid when the gallery mixes ages and angles).
    persons, owner = np.unique(np.asarray(ref_names, dtype=object), return_inverse=True)
    order = np.argsort(owner, kind='stable')
    refs_t = np.ascontiguousarray(refs[order].T)
    starts = np.flatnonzero(np.r_[True, np.diff(owner[order]) != 0])

    out: list[tuple[str | None, float]] = []
    for i in range(0, len(embeddings), CHUNK):
        per_person = np.maximum.reduceat(embeddings[i:i + CHUNK] @ refs_t, starts, axis=1)
        best = per_person.argmax(axis=1)
        rows = np.arange(len(per_person))
        score = per_person[rows, best]
        if per_person.shape[1] > 1:
            per_person[rows, best] = -np.inf
            second = per_person.max(axis=1)
        else:
            second = np.full(len(per_person), -1.0)
        ok = (score >= threshold) & (score - second >= margin)
        out += [(persons[b] if k else None, float(s)) for b, s, k in zip(best, score, ok)]
    return out


def cluster(embeddings: np.ndarray, threshold: float) -> list[np.ndarray]:
    """Greedy centroid clustering: a face joins the closest cluster whose centroid is at
    least ``threshold`` similar, else starts a new one. Linear in clusters per face, so it
    stays usable on 100k faces where a full similarity matrix would not fit.

    Returns the face indices of each cluster, largest first.
    """
    if not len(embeddings):
        return []
    # Buffers grow by doubling; only the first ``k`` rows are live clusters.
    sums = np.zeros((64, embeddings.shape[1]), dtype=np.float32)
    centroids = np.zeros_like(sums)
    labels = np.empty(len(embeddings), dtype=np.int64)
    k = 0

    for i, e in enumerate(embeddings):
        if k:
            sims = centroids[:k] @ e
            best = int(np.argmax(sims))
            if sims[best] >= threshold:
                labels[i] = best
                sums[best] += e
                centroids[best] = sums[best] / np.linalg.norm(sums[best])
                continue
        if k == len(sums):
            sums = np.vstack([sums, np.zeros_like(sums)])
            centroids = np.vstack([centroids, np.zeros_like(centroids)])
        labels[i] = k
        sums[k] = centroids[k] = e
        k += 1

    clusters = [np.flatnonzero(labels == c) for c in range(k)]
    return sorted(clusters, key=len, reverse=True)
