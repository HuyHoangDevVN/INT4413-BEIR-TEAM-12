# -*- coding: utf-8 -*-
"""Dense retrieval: encode corpus + queries bằng sentence-transformers, tìm top-k theo cosine similarity.

Cách chạy (nên có GPU/Colab):
    python run_dense.py --dataset scifact --model sentence-transformers/all-MiniLM-L6-v2
"""
import argparse

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from utils import (
    download_dataset,
    evaluate,
    filter_queries_by_qrels,
    load_corpus,
    load_qrels,
    load_queries,
    save_result,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=1000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[dense] device = {device}")

    folder = download_dataset(args.dataset)
    corpus = load_corpus(folder)
    queries = filter_queries_by_qrels(load_queries(folder), load_qrels(folder))
    print(f"[dense] corpus = {len(corpus):,} docs | queries = {len(queries)}")

    model = SentenceTransformer(args.model, device=device)

    doc_ids = list(corpus.keys())
    doc_texts = [f"{t} {x}".strip() for t, x in (corpus[d] for d in doc_ids)]
    doc_emb = model.encode(
        doc_texts, batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True
    )
    query_emb = model.encode(
        list(queries.values()), batch_size=args.batch_size, show_progress_bar=True, convert_to_numpy=True
    )

    # chuẩn hóa L2 rồi nhân ma trận = cosine similarity
    doc_emb = doc_emb / np.linalg.norm(doc_emb, axis=1, keepdims=True)
    query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
    scores = query_emb @ doc_emb.T  # (n_queries, n_docs)

    qrels = load_qrels(folder)
    qid_list = list(queries.keys())
    results = {}
    for i, qid in enumerate(qid_list):
        order = np.argsort(-scores[i])[: args.top_k]
        results[qid] = [doc_ids[j] for j in order]

    model_short = args.model.split("/")[-1]
    metrics = evaluate(results, qrels, f"{model_short}__{args.dataset}")
    save_result(args.dataset, model_short, metrics)


if __name__ == "__main__":
    main()
