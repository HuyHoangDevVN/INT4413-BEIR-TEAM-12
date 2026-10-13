# -*- coding: utf-8 -*-
"""Vẽ biểu đồ nDCG@10 so sánh các mô hình từ results/*.json -> results/chart_ndcg.png

Cách chạy:  python plot.py   (cần matplotlib: pip install matplotlib)
"""
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import RESULTS_DIR


def main():
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*.json"))):
        with open(path, encoding="utf-8") as f:
            rows.append(json.load(f))
    if not rows:
        print("Chưa có kết quả trong results/ — hãy chạy run_bm25.py / run_dense.py trước.")
        return

    datasets = sorted({r["dataset"] for r in rows})
    models = sorted({r["model"] for r in rows})

    x = range(len(datasets))
    width = 0.8 / max(len(models), 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, m in enumerate(models):
        vals = []
        for d in datasets:
            match = [r for r in rows if r["dataset"] == d and r["model"] == m]
            vals.append(match[0]["nDCG@10"] if match else 0)
        ax.bar([xi + i * width for xi in x], vals, width, label=m)

    ax.set_xticks([xi + width * (len(models) - 1) / 2 for xi in x])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("nDCG@10")
    ax.set_title("So sánh nDCG@10 giữa các mô hình (BEIR zero-shot)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    out = os.path.join(RESULTS_DIR, "chart_ndcg.png")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    print("[save]", out)


if __name__ == "__main__":
    main()
