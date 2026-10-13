# -*- coding: utf-8 -*-
"""Re-ranking: lấy top-100 của BM25 rồi xếp lại bằng cross-encoder.

Cách chạy:  python run_rerank.py --dataset scifact
"""
import argparse

import torch
from sentence_transformers import CrossEncoder

from utils import (
    build_bm25_index,
    bm25_retrieve,
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
    ap.add_argument("--model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--rerank-depth", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[rerank] device = {device}")

    folder = download_dataset(args.dataset)
    corpus = load_corpus(folder)
    queries = filter_queries_by_qrels(load_queries(folder), load_qrels(folder))

    print(f"[rerank] BM25 candidate generation trên {len(queries)} queries...")
    bm25, doc_ids = build_bm25_index(corpus)
    bm25_results = bm25_retrieve(bm25, doc_ids, queries, top_k=args.rerank_depth, log_every=0)

    ce = CrossEncoder(args.model, device=device, max_length=512)
    qid_list = list(queries.keys())
    results = {}
    print(f"[rerank] cross-encoder xếp lại top-{args.rerank_depth}...")
    for i, qid in enumerate(qid_list, 1):
        qtext = queries[qid]
        cand = bm25_results[qid][: args.rerank_depth]
        pairs = [(qtext, f"{corpus[d][0]} {corpus[d][1]}".strip()) for d in cand]
        scores = ce.predict(pairs, batch_size=args.batch_size)
        order = sorted(range(len(cand)), key=lambda j: scores[j], reverse=True)
        results[qid] = [cand[j] for j in order]
        if i % 50 == 0:
            print(f"  {i}/{len(qid_list)} queries")

    qrels = load_qrels(folder)
    model_short = args.model.split("/")[-1]
    metrics = evaluate(results, qrels, f"BM25+{model_short}__{args.dataset}")
    save_result(args.dataset, f"BM25+{model_short}", metrics)


if __name__ == "__main__":
    main()
