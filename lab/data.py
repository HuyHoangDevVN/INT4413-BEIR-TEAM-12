import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'scifact', 'nfcorpus', 'fiqa', 'arguana', 'trec-covid'}
# Official ArguAna has missing qrel documents (upstream issue #172). Keep all
# queries/qrels, never invent documents; only these audited IDs are tolerated.
ARGUANA_MISSING = {
    'test-free-speech-debate-yfsdfkhbwu-con03b', 'test-education-ufsdfkhbwu-con03b',
    'test-politics-dhwem-pro06b', 'test-science-sghwbdgmo-con03b', 'test-society-asfhwapg-con04b',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


def read_jsonl(folder, name):
    path = folder / (name + '.jsonl')
    if not path.exists():
        path = folder / (name + '.jsonl.gz')
    opener = gzip.open if path.suffix == '.gz' else open
    rows = {}
    with opener(path, 'rt', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            key = str(row['_id'])
            if key in rows:
                raise ValueError(f'Duplicate ID {key} in {path}')
            rows[key] = row
    if not rows:
        raise ValueError(f'Empty {path}')
    return rows


def load_dataset(name, split='test', data_dir=None, download=False):
    if name not in DATASETS:
        raise ValueError(f'Unsupported dataset: {name}')
    base = Path(data_dir or ROOT / 'data')
    folder = base / name
    if not folder.exists() and download:
        import requests
        base.mkdir(parents=True, exist_ok=True)
        archive = base / (name + '.zip.part')
        url = f'https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip'
        with requests.get(url, stream=True, timeout=300, verify=os.getenv('REQUESTS_CA_BUNDLE') or True) as response:
            response.raise_for_status()
            with archive.open('wb') as f:
                for chunk in response.iter_content(1024 * 1024):
                    f.write(chunk)
        with zipfile.ZipFile(archive) as z:
            for entry in z.infolist():
                target = (base / entry.filename).resolve()
                if not target.is_relative_to(base.resolve()):
                    raise ValueError('Unsafe archive path')
                if not target.is_relative_to(folder.resolve()):
                    raise ValueError('Archive contains files outside the requested dataset')
            z.extractall(base)
        archive.unlink()
    corpus_rows = read_jsonl(folder, 'corpus')
    query_rows = read_jsonl(folder, 'queries')
    corpus = {d: (r.get('title', ''), r.get('text', '')) for d, r in corpus_rows.items()}
    queries = {q: r['text'] for q, r in query_rows.items()}
    qrels = {}
    missing_documents = set()
    unjudged_count = 0
    with (folder / 'qrels' / f'{split}.tsv').open(encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        if next(reader) != ['query-id', 'corpus-id', 'score']:
            raise ValueError('Invalid qrels header')
        for row in reader:
            if len(row) != 3:
                raise ValueError('Malformed qrels row')
            q, d, score = row
            if q not in queries:
                raise ValueError(f'Unknown qrel ID: {q}/{d}')
            if d not in corpus:
                if name != 'arguana' or d not in ARGUANA_MISSING:
                    raise ValueError(f'Unknown qrel ID: {q}/{d}')
                missing_documents.add(d)
            rel = int(score)
            # TREC reserves -1 for unjudged; retain source label, evaluate as zero gain.
            if rel < -1 or d in qrels.get(q, {}):
                raise ValueError('Negative or duplicate qrel')
            unjudged_count += int(rel == -1)
            qrels.setdefault(q, {})[d] = rel
    if not qrels:
        raise ValueError('Empty qrels')
    queries = {q: t for q, t in queries.items() if q in qrels}
    metadata = {'corpus_checksum': digest(list(corpus.items())),
                                  'query_checksum': digest(list(queries.items())),
                                  'qrels_checksum': digest(qrels), 'dataset': name, 'split': split}
    if missing_documents:
        metadata['known_missing_qrel_documents'] = sorted(missing_documents)
        metadata['missing_qrel_policy'] = 'retain_all_queries_and_qrels; unretrievable_relevance_scores_zero'
    if unjudged_count:
        metadata['unjudged_qrels_count'] = unjudged_count
        metadata['unjudged_qrel_policy'] = 'retain_minus_one_source_label; zero_gain_and_not_relevant'
    return corpus, queries, qrels, metadata
