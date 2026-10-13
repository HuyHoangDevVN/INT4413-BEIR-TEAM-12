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

# TLS verification remains enabled. REQUESTS_CA_BUNDLE supports private CAs.
SSL_VERIFY = os.getenv("REQUESTS_CA_BUNDLE") or True


try:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # đang chạy trong notebook (Colab) thì lấy thư mục hiện tại
    BASE_DIR = os.path.abspath(os.getcwd())

DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATASET_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def download_dataset(name: str) -> str:
    from lab.data import load_dataset
    load_dataset(name, download=True)
    return os.path.join(DATA_DIR, name)


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
    from pathlib import Path
    from lab.data import read_jsonl
    return {d: (r.get("title", ""), r.get("text", "")) for d, r in read_jsonl(Path(folder), "corpus").items()}


def load_queries(folder: str) -> dict:
    from pathlib import Path
    from lab.data import read_jsonl
    return {q: r["text"] for q, r in read_jsonl(Path(folder), "queries").items()}


def load_qrels(folder: str, split: str = "test") -> dict:
    from pathlib import Path
    from lab.data import load_dataset
    path = Path(folder)
    return load_dataset(path.name, split, path.parent)[2]


def filter_queries_by_qrels(queries: dict, qrels: dict) -> dict:
    """Chỉ giữ các query có trong qrels (mỗi dataset BEIR chia train/dev/test)."""
    return {qid: text for qid, text in queries.items() if qid in qrels}


def legacy_exponential_dcg_at_k(rels, k: int) -> float:
    return sum((2 ** rel - 1) / math.log2(i + 2) for i, rel in enumerate(rels[:k]))


def legacy_exponential_ndcg_at_k(results: dict, qrels: dict, k: int = 10) -> float:
    """results: dict[qid] = danh sách doc_id xếp theo độ liên quan giảm dần."""
    scores = []
    for qid, ranked in results.items():
        gold = qrels.get(qid, {})
        if not gold:
            continue
        gained = [gold.get(doc_id, 0) for doc_id in ranked[:k]]
        ideal = sorted(gold.values(), reverse=True)[:k]
        idcg = legacy_exponential_dcg_at_k(ideal, k)
        scores.append(legacy_exponential_dcg_at_k(gained, k) / idcg if idcg > 0 else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def legacy_recall_at_k(results: dict, qrels: dict, k: int = 100) -> float:
    scores = []
    for qid, ranked in results.items():
        gold = qrels.get(qid, {})
        if not gold:
            continue
        hits = sum(1 for d in ranked[:k] if d in gold)
        scores.append(hits / len(gold))
    return sum(scores) / len(scores) if scores else 0.0


def evaluate(results: dict, qrels: dict, name: str) -> dict:
    from evaluation.metrics import evaluate as corrected_evaluate
    metrics = corrected_evaluate(results, qrels)
    print(name, metrics)
    return metrics


def save_result(dataset: str, model_name: str, metrics: dict):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    from uuid import uuid4
    path = os.path.join(RESULTS_DIR, f"{dataset}__{model_name}__{uuid4().hex[:8]}.json")
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


def ndcg_at_k(results, qrels, k=10):
    if k != 10:
        raise ValueError("V2 public metric is nDCG@10; use evaluation.metrics")
    from evaluation.metrics import evaluate
    return evaluate(results, qrels)["nDCG@10"]


def recall_at_k(results, qrels, k=100):
    if k not in (10, 100):
        raise ValueError("V2 supports Recall@10/100")
    from evaluation.metrics import evaluate
    return evaluate(results, qrels)[f"Recall@{k}"]


def dcg_at_k(rels, k):
    return sum(max(0, rel) / math.log2(i + 2) for i, rel in enumerate(rels[:k]))
