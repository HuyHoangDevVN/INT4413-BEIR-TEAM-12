# INT4413-BEIR-TEAM-12

Framework thực nghiệm BEIR của Nhóm 12, PTIT, học phần Tìm kiếm và Truy xuất thông tin. Project đối chứng baseline paper và mở rộng với các checkpoint retrieval/reranking hiện đại; không tuyên bố phát minh thuật toán mới.

## Phạm vi và kết quả

- Baseline paper: **Anserini BM25 k1=0.9/b=0.4**, **TAS-B**, **Anserini BM25 → MiniLM CE top-100**.
- Mở rộng: MiniLM dense, BGE-M3, Qwen3 Embedding 0.6B, hybrid RRF và các adapter reranker.
- SciFact/NFCorpus có đủ ba baseline và Qwen embedding/reranker top-20/50/100. [Bảng kết quả đã kiểm chứng](docs/results/README.md) và CSV kèm run ID, nguồn, checksum.
- FiQA/ArguAna/TREC-COVID dùng chung loader. Snapshot kết quả trong Git mới có một phần hệ thống trên ba dataset này; không gọi toàn matrix đã hoàn tất.
- Sáu JSON trong `results/` và `legacy/v1/results/` là kết quả V1 lịch sử chỉ có summary, chưa xác minh được bằng ranking. Không nhập số paper thành kết quả của nhóm.

Git chứa source, cấu hình, tests và bảng tổng hợp nhỏ. Corpus, model weights, embedding/index/cache, ranking lớn, môi trường ảo và gói bàn giao Word được giữ local, nằm trong `.gitignore`. Vì thiếu ranking trong Git, clone repo không tự xác minh lại các thí nghiệm lịch sử; cần artifact riêng.

## Cấu trúc

```text
beir_lab.py                    CLI run / benchmark / evaluate
lab/                           Loader, retrievers, model adapters, cache, runner
evaluation/                    Metric và kiểm định thống kê
configs/                       YAML thí nghiệm
run_paper_baseline.py           Anserini BM25 paper-era
run_paper_neural.py             TAS-B / Anserini → MiniLM CE100
run_qwen_embedding_quality.py  Qwen embedding quality-only
run_qwen_candidates.py          Qwen reranker trên candidate đã lưu
run_remaining_datasets.py       Hàng đợi matrix bổ sung trên Linux
run_remaining_cpu_baselines.py  Worker CPU, nhận session nguồn qua CLI
tools/                         Setup upstream, xuất summary, kiểm tra publish
tests/                         Unit/integration tests với dữ liệu nhỏ
docs/                          Audit, protocol, hướng dẫn báo cáo và kết quả
references/                    Số paper tham chiếu riêng
legacy/v1/                     Source V1 nguyên trạng
```

Các entrypoint V1 `run_bm25.py`, `run_dense.py`, `run_rerank.py` vẫn dùng được và gọi framework chung. Thư mục `legacy/v1/` chỉ để đối chiếu lịch sử: có evaluator/TLS behavior cũ, không phải entrypoint khuyến nghị.

## Cài môi trường riêng

Python **3.12 64-bit**; không dùng Python toàn máy nếu đang trỏ vào bản 32-bit. Không commit môi trường ảo.

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
.\.venv\Scripts\python.exe -m pip install -r requirements-neural.txt
# Optional: pytrec_eval và FAISS exact
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Linux:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt -r requirements-optional.txt
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Chọn wheel PyTorch CPU/CUDA phù hợp theo [hướng dẫn PyTorch](https://pytorch.org/get-started/locally/). GPU phần cứng không tự chứng minh runtime CUDA khả dụng. `requirements-lock.txt` là snapshot môi trường Windows đã dùng, không phải lock chung cho mọi máy.

## Chạy framework

Các ví dụ Linux dưới đây dùng `.venv/bin/python`; Windows thay bằng `.\.venv\Scripts\python.exe`.

```bash
# Dataset phải có tại data/scifact; --download chỉ dùng khi chủ động tải dữ liệu
.venv/bin/python beir_lab.py run --dataset scifact --retriever bm25 --download
.venv/bin/python beir_lab.py run --dataset nfcorpus --retriever bm25 --download
.venv/bin/python beir_lab.py run --config configs/qwen3.yaml --device cuda
.venv/bin/python beir_lab.py run --config configs/hybrid.yaml --candidate-k 50
.venv/bin/python beir_lab.py evaluate --run runs/RUN_ID
```

CLI ghi đè YAML rồi đến defaults; mỗi lượt có run ID mới. Model dùng cache nếu có, nhưng có thể tải weights nếu chưa có. Khi làm việc offline, chuẩn bị dataset/model trước. Không có ANN: NumPy chunked và optional FAISS IndexFlatIP đều exact; RRF dùng rank từ 1, mặc định k=60.

## Đối chứng baseline paper

Upstream được pin ở `52de18ae6f1ccc7426eb70895c0e43e1552ad493`. Đây là checkout paper-era đã chọn, không có bằng chứng đây là SHA tác giả dùng tạo bảng paper. Wrapper dùng dependency hiện tại và không tuyên bố runtime byte-identical năm 2021.

```bash
python3.12 -m venv .venv-paper
.venv-paper/bin/python -m pip install -r requirements-paper.txt
.venv-paper/bin/python tools/setup_paper_checkout.py --register
```

Windows dùng `py -3.12 -m venv .venv-paper`, sau đó `.\.venv-paper\Scripts\python.exe`. Setup chỉ clone checkout upstream và đăng ký trong venv được chọn; không reset checkout hiện có, không tải model/dataset. Không cài BEIR mới nhất thay cho source đã pin.

Tạo container BM25 riêng trên Linux/Docker Desktop:

```bash
docker run -d --name beir-paper-20211021 --cpus 2 --memory 8g --memory-swap 8g -e JAVA_TOOL_OPTIONS=-Xmx1536m -p 127.0.0.1:18021:8000 beir/pyserini-fastapi@sha256:c18cb7f60db4390978d3260bb9e80c797bf11e07e102a407323c7c5c804c83bf
.venv-paper/bin/python verify_paper_runtime.py
.venv-paper/bin/python run_paper_baseline.py --dataset scifact
.venv-paper/bin/python run_paper_baseline.py --dataset nfcorpus
```

Dữ liệu baseline paper phải chuẩn bị offline trước. Container không phải dataset/model downloader. Mỗi worker neural nhận thư mục output mới dưới `runs/`; reranker cần candidate run đúng dataset đã VERIFIED. Xem `--help` của worker tương ứng, [audit baseline](docs/PAPER_REPRODUCTION_AUDIT.md) và [cấu hình](configs/paper_bm25.yaml).

Hàng đợi bổ sung là script **Linux**:

```bash
.venv-paper/bin/python run_remaining_datasets.py
# Tiếp tục bằng phiên mới, reuse VERIFIED matching checksum:
.venv-paper/bin/python run_remaining_datasets.py --resume reports/remaining_datasets_SESSION
# CPU fallback: phải chọn rõ session chứa BM25 đã VERIFIED:
.venv-paper/bin/python run_remaining_cpu_baselines.py --source-session reports/remaining_datasets_SESSION --datasets arguana trec-covid --cpu-threads 8
```

Không khởi động hàng đợi trùng khi process cũ còn chạy. Các supervisor lịch sử khác có preset/run IDs của đợt SciFact/NFCorpus; không chạy chúng trên clone mới nếu chưa chuẩn bị đúng artifact nguồn.

## Metric, tài nguyên và giới hạn

- nDCG@10 gain tuyến tính; Recall@10/100, MRR@10, MAP@100 dùng relevance >0. Macro trên toàn query có qrels test; query thiếu ranking nhận 0. Gain mũ có tên riêng `legacy_exponential_ndcg@10`.
- Ranking chứa query/document/score/rank; từ chối document lặp và score không hữu hạn. Tie policy được ghi theo pipeline. `VERIFIED` yêu cầu đọc lại ranking/checksum/metric, không chỉ có config hay fixture.
- Reranker chỉ đổi thứ tự candidates. Recall@100 giữ nguyên với cùng 100 document; top-20/50 là output bị cắt, không suy ra top-100 chưa lưu.
- Baseline paper và Qwen top-50/100 trên server là quality-only, không có warm QPS. Laptop/server/CPU và latency end-to-end/reranker-only không được gộp để suy speedup.
- Hàng đợi giảm tải: batch 1, không nghỉ cố định trên server, bắt đầu ≤55°C, nghỉ 73→60°C, dừng 80°C, ngân sách phiên 24 giờ. Đây là ngưỡng vận hành của project; không tác động tiến trình người khác.
- Top-50 SciFact/top-100 NFCorpus là lựa chọn post-hoc từ kết quả test, không phải tối ưu trên dev độc lập. Không kết luận strict zero-shot/training-contamination-free hoặc cải thiện toàn BEIR.

## Kiểm thử và đóng góp

```bash
.venv-paper/bin/python -m pytest -q
.venv/bin/python beir_lab.py --help
.venv/bin/python tools/check_publish.py
```

Tests dùng fixture/corpus nhỏ, không chứng minh model thực chạy. Một số test optional sẽ skip nếu thiếu BEIR/FAISS/pytrec_eval/Torch. Trước khi commit, kiểm tra `git diff`, `git status` và chạy checks phù hợp; không đưa credentials hay artifact lớn vào Git.

## Tài liệu và nguồn

- [Audit mã V1](docs/CODE_AUDIT.md), [thiết kế thực nghiệm](docs/EXPERIMENT_DESIGN.md).
- [Kết quả và giới hạn](docs/results/README.md), [báo cáo Word](docs/REPORT_GUIDE.md), [hướng dẫn slide](docs/SLIDE_GUIDE.md).
- [Nguồn bên thứ ba](THIRD_PARTY_NOTICES.md), [BEIR paper](https://arxiv.org/abs/2104.08663), [BEIR upstream](https://github.com/beir-cellar/beir).

Các audit theo ngày giữ nguyên để thể hiện lịch sử; phát biểu tiến độ cũ không thay thế snapshot kết quả hoặc trạng thái live. Gói bằng chứng đầy đủ tại `report_handoff/` trên máy thực nghiệm không được commit vì chứa ranking lớn và metadata cục bộ.
