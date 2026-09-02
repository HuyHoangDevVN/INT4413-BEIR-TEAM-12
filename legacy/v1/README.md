# BEIR: Đánh giá zero-shot các mô hình truy xuất thông tin

Project bài tập lớn môn **Tìm kiếm và Truy xuất thông tin** — tái hiện benchmark BEIR (Thakur et al., NeurIPS 2021) trên 3 dataset nhỏ (SciFact, NFCorpus, FiQA-2018) với 3 tầng mô hình:

1. **BM25** — sparse retrieval cổ điển, không cần huấn luyện
2. **Dense retrieval** — bi-encoder từ `sentence-transformers`
3. **Re-ranking** — cross-encoder xếp lại top-100 của BM25

Độ đo: **nDCG@10** (giống paper) và Recall@100.

## ⚠️ Điểm quan trọng nhất cần hiểu

**BEIR là benchmark ZERO-SHOT nên KHÔNG CÓ bước train/fine-tune.** Ý nghĩa của benchmark là: lấy mô hình đã huấn luyện sẵn (trên MS MARCO) áp thẳng sang domain khác, đo xem nó tổng quát tốt thế nào. Code này là pipeline *đánh giá*, không phải huấn luyện.

## Cấu trúc

```
BEIR-project/
├── utils.py            # tải dataset, đọc corpus/queries/qrels, nDCG@10, Recall@100
├── run_bm25.py         # BM25 (chạy được trên CPU, cả máy cá nhân lẫn Colab)
├── run_dense.py        # dense retrieval (nên chạy Colab GPU)
├── run_rerank.py       # cross-encoder re-ranking (nên chạy Colab GPU)
├── collect_results.py  # tổng hợp results/ -> summary.csv + summary.md
├── requirements.txt
├── BEIR_colab.ipynb    # notebook chạy toàn bộ trên Google Colab
└── results/            # kết quả .json, tự sinh khi chạy
```

## Cách chạy trên Google Colab (khuyên dùng)

1. Vào [colab.research.google.com](https://colab.research.google.com) → **File → Upload notebook** → chọn `BEIR_colab.ipynb`
2. **Runtime → Change runtime type → T4 GPU** (chỉ cần cho dense/re-rank; BM25 chạy CPU cũng được)
3. Chạy lần lượt từng cell — notebook sẽ cài thư viện, tải dataset, chạy cả 3 mô hình, tổng hợp kết quả.
4. Kết quả nằm ở `results/summary.csv` — tải về máy để đưa vào báo cáo.

Thời lượng ước tính trên T4: SciFact ~3 phút, NFCorpus ~2 phút, FiQA ~15–25 phút.

## Cách chạy trên máy cá nhân (Windows)

```bash
pip install -r requirements.txt

python run_bm25.py --dataset scifact
python run_dense.py --dataset nfcorpus --model sentence-transformers/all-MiniLM-L6-v2
python run_rerank.py --dataset scifact
python collect_results.py
```

Dataset tự tải về `data/` (~200–400 MB mỗi dataset), kết quả ghi vào `results/`.

## Dataset đề xuất

| Dataset | Domain | Số docs | Đặc điểm |
|---|---|---|---|
| scifact | Khoa học (fact-checking) | 5.183 | nhỏ nhất, chạy nhanh |
| nfcorpus | Y sinh | 3.633 | nhỏ, nDCG thấp — cho thấy dense khó |
| fiqa | Tài chính (QA) | 57.638 | vừa, dense thắng BM25 rõ |

Muốn thêm dataset to hơn: `trec-covid` (~171K docs), `hotpotqa`, `scidocs`.

## Số liệu tham chiếu từ paper gốc (nDCG@10)

| Dataset | BM25 | DPR | ANCE | TAS-B |
|---|---|---|---|---|
| SciFact | 0.665 | 0.646 | 0.505 | 0.656 |
| NFCorpus | 0.318 | 0.317 | 0.310 | 0.315 |
| FiQA | 0.236 | 0.225 | 0.296 | 0.358 |

Kết quả chạy của bạn có thể lệch ±0.02–0.05 so với paper do khác bản thư viện, tham số BM25, cách tiền xử lý.

## Kết quả đã chạy thử với project này (BM25)

| Dataset | nDCG@10 (chạy) | nDCG@10 (paper) | Recall@100 |
|---|---|---|---|
| SciFact | 0.6523 | 0.665 | 0.8731 |
| NFCorpus | 0.3076 | 0.318 | 0.2368 |

→ Lệch ~0.01–0.013 so với paper: mức tái hiện đạt yêu cầu cho bài tập môn học.

## Tài liệu

- Paper: [BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models](https://arxiv.org/abs/2104.08663)
- Repo gốc: https://github.com/beir-cellar/beir
