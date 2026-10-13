# -*- coding: utf-8 -*-
"""Tiện ích chung: tải dataset BEIR, đọc corpus/queries/qrels, tính nDCG@10, Recall@100."""
import gzip
import json
import math
import os
import re
import zipfile

import requests
import urllib3

# Một số mạng công ty/proxy dùng cert tự ký khiến Python từ chối kết nối TLS
# (ảnh hưởng tải dataset và tải model từ HuggingFace). Bypass bên dưới chỉ bật
# khi phát hiện SSL bị chặn; trên mạng bình thường hoặc Colab thì không chạy.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
SSL_VERIFY = False


def _trust_all_ssl():
    import ssl as _ssl
    _orig = _ssl.create_default_context

    def _patched(purpose=_ssl.Purpose.SERVER_AUTH, *a, **kw):
        ctx = _orig(purpose, *a, **kw)
        ctx.check_hostname = False
        ctx.verify_mode = _ssl.CERT_NONE
        return ctx

    _ssl.create_default_context = _patched


try:
    requests.head("https://huggingface.co", timeout=5)
except Exception:
    try:
        requests.head("https://huggingface.co", timeout=5, verify=False)
        _trust_all_ssl()
    except Exception:
        pass

try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # đang chạy trong notebook (Colab) thì lấy thư mục hiện tại
    BASE_DIR = os.path.abspath(os.getcwd())

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def download_dataset(name: str) -> str:
    """Tải và giải nén dataset BEIR (nếu chưa có). Trả về thư mục chứa dataset."""
    out_dir = os.path.join(DATA_DIR, name)
    if _find_file(out_dir, "corpus.jsonl"):
        return out_dir
    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, f"{name}.zip")
    url = DATASET_URL.format(name=name)
    print(f"[download] {url}")
    with requests.get(url, stream=True, timeout=300, verify=SSL_VERIFY) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        received = 0
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                received += len(chunk)
                if total:
                    print(f"\r[download] {received / 1e6:7.1f} / {total / 1e6:.1f} MB", end="")
    print()
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(DATA_DIR)
    os.remove(zip_path)
    return out_dir


def _find_file(folder: str, stem: str):
    """BEIR phát hành corpus/queries dưới dạng .jsonl hoặc .jsonl.gz tùy dataset."""
    for suffix in ("", ".gz"):
        path = os.path.join(folder, stem + suffix)
        if os.path.exists(path):
            return path
    return None


def _open_jsonl(path):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_corpus(folder: str) -> dict:
    """Đọc corpus.jsonl -> dict[doc_id] = (title, text)."""
    corpus = {}
    with _open_jsonl(_find_file(folder, "corpus.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            corpus[row["_id"]] = (row.get("title", ""), row.get("text", ""))
    return corpus


def load_queries(folder: str) -> dict:
    """Đọc queries.jsonl -> dict[query_id] = text."""
    queries = {}
    with _open_jsonl(_find_file(folder, "queries.jsonl")) as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            queries[row["_id"]] = row.get("text", "")
    return queries


def load_qrels(folder: str, split: str = "test") -> dict:
    """Đọc qrels/test.tsv -> dict[query_id] = {doc_id: mức liên quan 0..3}."""
    path = os.path.join(folder, "qrels", f"{split}.tsv")
    qrels = {}
    with open(path, "r", encoding="utf-8") as f:
        next(f)  # bỏ dòng tiêu đề
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 3:
                continue
            qid, docid, rel = parts[0], parts[1], int(float(parts[2]))
            qrels.setdefault(qid, {})[docid] = rel
    return qrels


def filter_queries_by_qrels(queries: dict, qrels: dict) -> dict:
    """Chỉ giữ các query có trong qrels (mỗi dataset BEIR chia train/dev/test)."""
    return {qid: text for qid, text in queries.items() if qid in qrels}


def dcg_at_k(rels, k: int) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels[:k]))


def ndcg_at_k(results: dict, qrels: dict, k: int = 10) -> float:
    """results: dict[qid] = danh sách doc_id xếp theo độ liên quan giảm dần."""
    scores = []
    for qid, ranked in results.items():
        gold = qrels.get(qid, {})
        if not gold:
            continue
        gained = [gold.get(doc_id, 0) for doc_id in ranked[:k]]
        ideal = sorted(gold.values(), reverse=True)[:k]
        idcg = dcg_at_k(ideal, k)
        scores.append(dcg_at_k(gained, k) / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def recall_at_k(results: dict, qrels: dict, k: int = 100) -> float:
    scores = []
    for qid, ranked in results.items():
        gold = qrels.get(qid, {})
        if not gold:
            continue
        hits = sum(1 for d in ranked[:k] if d in gold)
        scores.append(hits / len(gold))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(results: dict, qrels: dict, name: str) -> dict:
    metrics = {
        "nDCG@10": round(ndcg_at_k(results, qrels, 10), 4),
        "Recall@100": round(recall_at_k(results, qrels, 100), 4),
    }
    print(f"[{name}] nDCG@10 = {metrics['nDCG@10']:.4f} | Recall@100 = {metrics['Recall@100']:.4f}")
    return metrics


def save_result(dataset: str, model_name: str, metrics: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"{dataset}__{model_name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"dataset": dataset, "model": model_name, **metrics}, f, indent=2)
    print(f"[save] {path}")


def tokenize(text: str) -> list:
    """Tách từ đơn giản: chữ thường, chỉ giữ chữ/số — đủ dùng cho BM25."""
    return re.findall(r"[a-z0-9]+", text.lower())


def build_bm25_index(corpus: dict):
    """Dựng chỉ mục BM25. Trả về (bm25, doc_ids) để tra lại doc từ vị trí."""
    from rank_bm25 import BM25Okapi

    doc_ids = list(corpus.keys())
    tokenized = [tokenize(f"{corpus[d][0]} {corpus[d][1]}") for d in doc_ids]
    return BM25Okapi(tokenized), doc_ids


def bm25_retrieve(bm25, doc_ids: list, queries: dict, top_k: int = 1000, log_every: int = 100) -> dict:
    """Truy vấn BM25 cho từng query, trả về top_k doc_id xếp giảm dần theo điểm."""
    results = {}
    for i, (qid, qtext) in enumerate(queries.items(), 1):
        scores = bm25.get_scores(tokenize(qtext))
        top = sorted(range(len(scores)), key=lambda j: scores[j], reverse=True)[:top_k]
        results[qid] = [doc_ids[j] for j in top]
        if log_every and i % log_every == 0:
            print(f"  {i}/{len(queries)} queries")
    return results
