import json
import os
import uuid
import hashlib
from pathlib import Path
import numpy as np


def file_digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


def finalized_artifact_hashes(root):
    """Exclude in-flight jobs when independent CPU/GPU sessions overlap."""
    root=Path(root); result={}
    terminal={'VERIFIED','EXECUTION_ERROR','NOT_RUN_RESOURCE_LIMIT','RUN_FAILED','MODEL_DOWNLOAD_FAILURE'}
    for folder in (root/'runs').iterdir():
        if folder.is_file():
            result[str(folder.relative_to(root))]=file_digest(folder)
        elif (folder/'status.json').exists() and json.loads((folder/'status.json').read_text())['status'] in terminal:
            result.update({str(p.relative_to(root)):file_digest(p) for p in folder.rglob('*') if p.is_file()})
    result.update({str(p.relative_to(root)):file_digest(p) for p in (root/'results').rglob('*') if p.is_file()})
    return result


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


class EmbeddingCache:
    def __init__(self, folder):
        self.folder = Path(folder)

    def load(self, manifest, ids):
        try:
            stored = json.loads((self.folder / 'manifest.json').read_text(encoding='utf-8'))
            if any(stored.get(k) != v for k, v in manifest.items()):
                return None
            if json.loads((self.folder / 'doc_ids.json').read_text()) != ids:
                return None
            if file_digest(self.folder / 'corpus.npy') != stored['file_checksum']:
                return None
            matrix = np.load(self.folder / 'corpus.npy', mmap_mode='r')
            if list(matrix.shape) != stored['shape'] or matrix.shape[0] != len(ids):
                return None
            if matrix.dtype != np.float32 or not np.isfinite(matrix).all():
                return None
            return matrix
        except (OSError, ValueError, KeyError):
            return None

    def save(self, manifest, ids, matrix):
        self.folder.mkdir(parents=True, exist_ok=True)
        matrix = np.asarray(matrix, dtype=np.float32)
        temp = self.folder / ('corpus.' + uuid.uuid4().hex + '.tmp')
        with temp.open('wb') as f:
            np.save(f, matrix)
        os.replace(temp, self.folder / 'corpus.npy')
        atomic_json(self.folder / 'doc_ids.json', ids)
        atomic_json(self.folder / 'manifest.json', {**manifest, 'dimension': matrix.shape[1], 'shape': list(matrix.shape),
                                                    'file_checksum': file_digest(self.folder / 'corpus.npy')})


def directory_size(folder):
    return sum(p.stat().st_size for p in Path(folder).rglob('*') if p.is_file())


class CandidateCache:
    def __init__(self, path):
        self.path = Path(path)

    def load(self, manifest, queries, corpus):
        from evaluation.metrics import evaluate
        from .data import digest
        try:
            value = json.loads(self.path.read_text())
            rows = value['ranking']
            if value['manifest'] != manifest or value['checksum'] != digest(rows) or set(rows) != set(queries):
                return None
            evaluate(rows, {})  # Reject duplicates and non-finite scores.
            for ranking in rows.values():
                if any(r['document_id'] not in corpus or r['rank'] != i for i, r in enumerate(ranking, 1)):
                    return None
            return rows
        except (OSError, ValueError, KeyError):
            return None

    def save(self, manifest, rows):
        from .data import digest
        atomic_json(self.path, {'manifest': manifest, 'ranking': rows, 'checksum': digest(rows)})
