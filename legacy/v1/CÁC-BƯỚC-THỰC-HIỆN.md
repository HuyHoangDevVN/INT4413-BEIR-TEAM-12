# 📋 CÁC BƯỚC THỰC HIỆN BÀI TẬP LỚN BEIR

> Kiểm danh sách này theo thứ tự. Đánh dấu ✅ vào từng mục khi xong.
> Project: `D:\BEIR-project` | Độ đo: nDCG@10, Recall@100 | 3 tầng mô hình: BM25 → Dense → Re-rank

---

## TỔNG QUAN: AI LÀM GÌ Ở ĐÂU

| Nơi làm | Việc | Vì sao |
|---|---|---|
| **Máy cá nhân (ổ D)** | BM25, tổng hợp kết quả, vẽ biểu đồ, viết báo cáo/slide | BM25 không cần GPU — đã chạy thử thành công |
| **Google Colab** | Dense retrieval + Cross-encoder re-ranking | Cần GPU T4 miễn phí của Colab |

---

## GIAI ĐOẠN 0 — HIỂU CHỦ ĐỀ (1–2 ngày)

- [ ] **Bước 0.1.** Đọc paper BEIR lượt 1 (chỉ đọc): mở https://arxiv.org/abs/2104.08663 → tải PDF
  - Chú ý 3 thứ: (1) BEIR gồm 18 dataset / 13 domain, (2) "zero-shot" = không fine-tune trên dataset đích, (3) BM25 rất khó thắng trong zero-shot.
- [ ] **Bước 0.2.** Đọc paper lượt 2 kỹ hơn: Bảng 1 (18 dataset), Bảng 2 (kết quả nDCG@10), mục 4 (phân tích vì sao dense thua BM25).
- [ ] **Bước 0.3.** Đọc thêm 2 bài nền (ít nhất đọc abstract + ý chính):
  - BM25: Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond"
  - Dense retrieval: Karpukhin et al., "Dense Passage Retrieval for Open-Domain Question Answering" (https://arxiv.org/abs/2004.04906)

---

## GIAI ĐOẠN 1 — CHẠY BM25 TRÊN MÁY CÁ NHÂN (đã xong sẵn 1 phần ✅)

> Project đã có sẵn kết quả SciFact + NFCorpus cho BM25. Bạn chỉ cần chạy lại để hiểu + thêm dataset nếu muốn.

- [x] **Bước 1.1.** Cài thư viện (đã cài trên máy):
  ```
  cd /d D:\BEIR-project
  pip install -r requirements.txt
  ```
- [ ] **Bước 1.2.** Chạy BM25 lần đầu do chính bạn (giúp hiểu pipeline):
  ```
  python run_bm25.py --dataset scifact
  ```
  → Kết quả: `results\scifact__BM25.json`, in ra nDCG@10 ≈ **0.6523** (paper: 0.665)
- [ ] **Bước 1.3.** Chạy nốt dataset còn lại:
  ```
  python run_bm25.py --dataset nfcorpus
  ```
  → nDCG@10 ≈ **0.3076** (paper: 0.318)
- [ ] **Bước 1.4. (tùy chọn, nên làm)** Thêm dataset thứ 3:
  ```
  python run_bm25.py --dataset fiqa
  ```
  → chờ khoảng 5–15 phút (corpus 57K docs). Kỳ vọng nDCG@10 ≈ 0.22–0.24.

**Đầu ra giai đoạn này:** 2–3 file `.json` trong `results/` + điểm BM25 để đưa vào báo cáo.

---

## GIAI ĐOẠN 2 — CHẠY DENSE + RE-RANK TRÊN GOOGLE COLAB (bắt buộc dùng Colab)

> File cần: `D:\BEIR-project\BEIR_colab.ipynb` (đã viết sẵn, 10 cell).

- [ ] **Bước 2.1.** Vào https://colab.research.google.com → đăng nhập Google
- [ ] **Bước 2.2.** **File → Upload notebook** → chọn `BEIR_colab.ipynb` từ ổ D
- [ ] **Bước 2.3.** **Runtime → Change runtime type → chọn T4 GPU → Save**
- [ ] **Bước 2.4.** Chạy **Cell 1** (cài thư viện) → chờ 1–2 phút, phải thấy `GPU: Tesla T4`
- [ ] **Bước 2.5.** Chạy **Cell 2, 3, 4** (thư mục + hàm dùng chung + tải dataset scifact/nfcorpus)
- [ ] **Bước 2.6.** Chạy **Cell 5** — BM25 (chạy lại cho đủ bộ kết quả trong 1 nơi)
- [ ] **Bước 2.7.** Chạy **Cell 6** — Dense retrieval (encode corpus, tiến độ chạy ~2–5 phút cho 2 dataset nhỏ)
  → ghi lại nDCG@10 của `all-MiniLM-L6-v2` (kỳ vọng: SciFact ~0.50–0.64, NFCorpus ~0.30–0.32)
- [ ] **Bước 2.8.** Chạy **Cell 7** — Cross-encoder re-rank (kỳ vọng nDCG@10 SciFact ~0.68–0.74 — thường cao nhất)
- [ ] **Bước 2.9.** Chạy **Cell 8** — tổng hợp bảng + xuất `results/summary.csv`
- [ ] **Bước 2.10.** Chạy **Cell 9** — tự tải `summary.csv` về máy
- [ ] **Bước 2.11. (tùy chọn)** Muốn thêm FiQA: sửa Cell 4 thành `DATASETS = ['scifact', 'nfcorpus', 'fiqa']` rồi chạy lại Cell 4→8 (fiqa mất ~15–25 phút trên GPU)
- [ ] **Bước 2.11b.** (nếu dùng Drive) Mount Google Drive để lưu kết quả, thêm vào đầu notebook:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  ```
  Lưu ý: Colab **xóa dữ liệu mỗi phiên** (nếu không mount Drive) — lần chạy sau phải tải dataset lại.

**Đầu ra giai đoạn này:** `summary.csv` chứa đủ: BM25, all-MiniLM-L6-v2 (dense), BM25+CE (re-rank) × 2–3 dataset.

---

## GIAI ĐOẠN 3 — TỔNG HỢP SỐ LIỆU + BIỂU ĐỒ TRÊN MÁY

- [ ] **Bước 3.1.** Copy các file `*.json` từ Colab (Cell 9 chỉ tải summary.csv; nếu muốn đủ json, tải từng file trong tab Files bên trái Colab) vào `D:\BEIR-project\results\`
  - *Lưu ý:* file `scifact__BM25.json` và `nfcorpus__BM25.json` đã có sẵn trên máy từ trước.
- [ ] **Bước 3.2.** Vẽ biểu đồ:
  ```
  cd /d D:\BEIR-project
  python plot.py
  ```
  → ra `results\chart_ndcg.png` (biểu đồ cột so sánh các mô hình) — đưa thẳng vào báo cáo + slide.

---

## GIAI ĐOẠN 4 — VIẾT BÁO CÁO WORD (~20–30 trang)

> Tải template trước: https://templatecanthieu.vn (hoặc template của trường/khoa). Font thường: Times New Roman 13, giãn dòng 1.5.

- [ ] **Bước 4.1.** Chương 1 — Giới thiệu (2–3 trang): bài toán IR là gì, vì sao cần đánh giá zero-shot, mục tiêu bài tập
- [ ] **Bước 4.2.** Chương 2 — Cơ sở lý thuyết (6–8 trang) *← phần chiếm điểm nhiều nhất:*
  - 2.1. Bài toán truy xuất: corpus / query / qrel
  - 2.2. Độ đo: nDCG@10 (viết rõ công thức DCG = Σ (2^rel − 1)/log₂(i+1), giải thích IDCG), Recall@k
  - 2.3. Sparse retrieval: TF-IDF → BM25 (công thức, ý nghĩa k1, b)
  - 2.4. Dense retrieval: bi-encoder, embedding, cosine similarity
  - 2.5. Cross-encoder re-ranking
  - 2.6. Zero-shot evaluation là gì
- [ ] **Bước 4.3.** Chương 3 — BEIR (3–4 trang): 18 dataset / 13 domain (kể tên nhóm), các loại task (fact-checking, QA, argument retrieval, duplicate question, entity retrieval), pipeline đánh giá
- [ ] **Bước 4.4.** Chương 4 — Thí nghiệm (3–4 trang): mô tả setup — dataset chọn (bảng), mô hình (BM25 qua rank_bm25, all-MiniLM-L6-v2, cross-encoder ms-marco-MiniLM-L-6-v2), phần cứng (Colab T4), quy trình (ảnh chụp code/notebook)
- [ ] **Bước 4.5.** Chương 5 — Kết quả & thảo luận (4–5 trang):
  - Bảng nDCG@10: chạy của bạn vs paper (lấy từ summary.csv)
  - Chèn `chart_ndcg.png`
  - Phân tích: dense có thắng BM25 không? vì sao? re-rank cải thiện bao nhiêu? so sánh xu hướng với paper
- [ ] **Bước 4.6.** Chương 6 — Kết luận (1–2 trang): rút ra bài học về tổng quát hóa của dense retrieval, hạn chế (số dataset ít, 1 model dense), hướng phát triển (thêm dataset, hybrid retrieval)
- [ ] **Bước 4.7.** Tài liệu tham khảo (chuẩn IEEE):
  1. N. Thakur et al., "BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models," NeurIPS 2021.
  2. V. Karpukhin et al., "Dense Passage Retrieval…," EMNLP 2020.
  3. S. Robertson, H. Zaragoza, "The Probabilistic Relevance Framework: BM25 and Beyond," 2009.
  4. N. Reimers, I. Gurevych, "Sentence-BERT," EMNLP 2019.
- [ ] **Bước 4.8.** Danh sách bảng + hình

**Đầu ra giai đoạn này:** `BaoCao_BEIR.docx` hoàn chỉnh.

---

## GIAI ĐOẠN 5 — SLIDE THUYẾT TRÌNH (~12–15 slide)

- [ ] **Bước 5.1.** Dàn ý slide:
  1. Bìa (tên bài, môn, tên nhóm)
  2. Bài toán & động cơ (vì sao cần zero-shot benchmark)
  3. Zero-shot evaluation là gì
  4. Tổng quan BEIR — 18 dataset / 13 domain (sơ đồ)
  5. 3 loại mô hình đánh giá (BM25 / dense / cross-encoder) — mỗi cái 1 slide
  6. Thiết kế thí nghiệm (dataset, phần cứng, quy trình)
  7. Kết quả 1: bảng nDCG@10
  8. Kết quả 2: biểu đồ (chèn chart_ndcg.png)
  9. Nhận xét: xu hướng giống/khác paper, nguyên nhân
  10. Hạn chế & hướng phát triển
  11. Kết luận
- [ ] **Bước 5.2.** Quy tắc: mỗi slide ≤ 6 dòng chữ, số liệu + hình là chính; chữ dài để trong báo cáo
- [ ] **Bước 5.3.** Tập nói thử, mỗi người ~7–10 phút

**Đầu ra:** `Slide_BEIR.pptx`.

---

## ⏱ LỘ TRÌNH ĐỀ XUẤT (3 tuần)

| Tuần | Việc | Giai đoạn |
|---|---|---|
| 1 | Đọc paper, viết Chương 1–3 | 0, 4.1–4.3 |
| 2 | Chạy BM25 (máy) + Colab dense/re-rank, thu số liệu | 1, 2 |
| 3 | Biểu đồ, Chương 4–5, hoàn thiện báo cáo + slide | 3, 4, 5 |

---

## ⚠️ LỖI THƯỜNG GẶP

| Triệu chứng | Cách xử lý |
|---|---|
| `SSLError: self-signed certificate` khi tải dataset trên máy | Đã được xử lý sẵn trong `utils.py` (SSL_VERIFY=False) |
| Colab hết GPU (thông báo "cannot connect to GPU runtime") | Đổi sang buổi khác, hoặc xài runtime "CPU" — dense chậm hơn nhưng chạy được với dataset nhỏ |
| Colab mất dữ liệu khi đóng | Bình thường — dataset/kết quả chỉ sống trong 1 phiên; tải file về hoặc mount Drive |
| FiQA chạy lâu trên máy cá nhân | Chỉ chạy BM25 cho fiqa ở máy; dense/re-rank cho fiqa để trên Colab |
| Kết quả lệch nhiều so với paper | Kiểm tra đã dùng đúng split `test` cho qrels; lệch ≤0.05 là bình thường |
