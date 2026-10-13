# -*- coding: utf-8 -*-
"""Tổng hợp mọi kết quả trong thư mục results/ thành bảng CSV + Markdown.

Cách chạy:  python collect_results.py
"""
import csv
import glob
import json
import os

from utils import RESULTS_DIR

PAPER_NDCG = {
    # nDCG@10 báo cáo trong bài BEIR (Thakur et al., NeurIPS 2021), Bảng 2
    ("scifact", "bm25"): 0.665,
    ("scifact", "dpr"): 0.646,
    ("scifact", "ance"): 0.505,
    ("scifact", "tas-b"): 0.656,
    ("nfcorpus", "bm25"): 0.318,
    ("nfcorpus", "dpr"): 0.317,
    ("nfcorpus", "ance"): 0.310,
    ("nfcorpus", "tas-b"): 0.315,
    ("fiqa", "bm25"): 0.236,
    ("fiqa", "dpr"): 0.225,
    ("fiqa", "ance"): 0.296,
    ("fiqa", "tas-b"): 0.358,
    ("trec-covid", "bm25"): 0.656,
    ("trec-covid", "dpr"): 0.359,
    ("trec-covid", "ance"): 0.654,
    ("trec-covid", "tas-b"): 0.479,
}


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))

    # nhóm theo dataset
    by_dataset = {}
    for r in rows:
        by_dataset.setdefault(r["dataset"], []).append(r)

    lines = []
    for dataset, entries in by_dataset.items():
        lines.append(f"\n### {dataset}\n")
        lines.append("| Mô hình | nDCG@10 (chạy) | nDCG@10 (paper) | Recall@100 (chạy) |")
        lines.append("|---|---|---|---|")
        for r in entries:
            paper = PAPER_NDCG.get((dataset.lower(), r["model"].split("+")[0].lower()), "-")
            lines.append(f"| {r['model']} | {r['nDCG@10']:.4f} | {paper} | {r.get('Recall@100', '-')} |")

    table_md = "\n".join(lines)
    print(table_md)

    csv_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset", "model", "nDCG@10", "Recall@100", "nDCG@10_paper"])
        for dataset, entries in by_dataset.items():
            for r in entries:
                paper = PAPER_NDCG.get((dataset.lower(), r["model"].split("+")[0].lower()), "")
                writer.writerow([dataset, r["model"], r["nDCG@10"], r.get("Recall@100", ""), paper])
    print(f"\n[save] {csv_path}")

    with open(os.path.join(RESULTS_DIR, "summary.md"), "w", encoding="utf-8") as f:
        f.write(table_md)


if __name__ == "__main__":
    main()
