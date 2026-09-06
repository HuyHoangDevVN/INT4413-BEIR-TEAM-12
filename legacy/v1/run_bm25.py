# -*- coding: utf-8 -*-
"""Chạy BM25 (sparse retrieval) trên một dataset BEIR.

Cách chạy:  python run_bm25.py --dataset scifact
"""
import argparse

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
    ap.add_argument("--dataset", required=True, help="scifact | nfcorpus | fiqa | trec-covid | ...")
    ap.add_argument("--top-k", type=int, default=1000, help="số kết quả giữ lại mỗi query")
    args = ap.parse_args()

    folder = download_dataset(args.dataset)
    corpus = load_corpus(folder)
    queries = filter_queries_by_qrels(load_queries(folder), load_qrels(folder))

    print(f"[bm25] corpus = {len(corpus):,} docs | queries = {len(queries)}")
    bm25, doc_ids = build_bm25_index(corpus)
    results = bm25_retrieve(bm25, doc_ids, queries, top_k=args.top_k)

    qrels = load_qrels(folder)
    metrics = evaluate(results, qrels, f"BM25__{args.dataset}")
    save_result(args.dataset, "BM25", metrics)


if __name__ == "__main__":
    main()
