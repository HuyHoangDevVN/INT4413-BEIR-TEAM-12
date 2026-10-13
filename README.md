# BEIR Lab V2

## Hoàn tất baseline neural trên server — 12/10/2026

Đã chạy TAS-B và Anserini BM25 → MiniLM-L6-v2 CE top-100 trên SciFact/NFCorpus, đủ query và VERIFIED. Ba baseline BM25/TAS-B/CE trên mỗi dataset làm tròn nDCG@10 khớp Table 2 paper. Xem [bảng đầy đủ với Qwen](reports/paper_reproduction/full_baselines_20261012/comparison.md).

Server: Ubuntu/A30, Python 3.12.7/PyTorch 2.6.0+cu124, FP32, context 512, batch 1. Theo yêu cầu mới, bỏ nghỉ cố định trên server nhưng giữ giám sát nhiệt: bắt đầu <=55°C, tạm nghỉ từ 73 tới 60°C, dừng 80°C. Phiên hoàn chỉnh khoảng 11 phút, một lần nghỉ nhiệt, không chạm 80°C. Chỉ đo chất lượng một lượt; không có warm latency/QPS. Qwen giữ nguyên artifact laptop có nghỉ 0,1 giây/batch, không so latency khác thiết bị/protocol.

Worker: `run_paper_neural.py`; supervisor: `run_paper_neural_session.py`. CE dùng candidate Anserini đã VERIFIED, giữ nguyên top-100 kể cả query no-match rỗng. Cấu hình/runtime/model revision và checksum nằm trong run artifacts. Checkout upstream giữ nguyên source; shim ngoài checkout nạp file SentenceBERT trực tiếp để tránh TensorFlow không dùng và chuyển tên tham số predict cho SentenceTransformers hiện tại.

## Đối chứng paper BEIR 2021

Baseline paper chính là **Anserini BM25 k1=0.9/b=0.4**, title/contents riêng, không phải `rank_bm25` V1 hay Elasticsearch. Đã chạy upstream paper-era trên SciFact/NFCorpus; nDCG@10 làm tròn khớp Table 2 (0.665/0.325). Xem [so sánh với Qwen](reports/paper_reproduction/comparison.md), [audit](docs/PAPER_REPRODUCTION_AUDIT.md) và [cấu hình chốt](configs/paper_bm25.yaml). SHA đã chốt không được chứng minh là SHA thực nghiệm tác giả; wrapper dùng dependency hiện tại, không tuyên bố runtime byte-identical năm 2021.

Checkout `upstream/beir` nguyên trạng ở `52de18ae6f1ccc7426eb70895c0e43e1552ad493`. `.venv-paper` riêng dùng cùng thư viện của `.venv` qua file `.pth`, cộng Elasticsearch 7.17.12 chỉ để import evaluator cũ. Không thay Python hay package toàn máy. Docker của tác giả được pin digest, Pyserini 0.12.0/Java 11.

Chạy lại CPU baseline bằng container riêng (chỉ một dataset tại một thời điểm):

```powershell
docker start beir-paper-20211021
.\.venv-paper\Scripts\python.exe run_paper_baseline.py --dataset scifact
.\.venv-paper\Scripts\python.exe run_paper_baseline.py --dataset nfcorpus
.\.venv-paper\Scripts\python.exe compare_paper_qwen.py
docker stop beir-paper-20211021
```

Container đã tạo với localhost:18021, 2 CPU/3 GiB RAM, Java heap 1536 MiB. Qwen giữ nguyên các run giảm tải cũ; bảng so sánh dùng quality cùng qrels, không so latency với CPU/Docker. Chưa chạy các baseline neural của paper như TAS-B hay BM25+CE top-100 trong lượt đối chứng BM25 này.

## What is BEIR?
BEIR là benchmark đánh giá retrieval trên nhiều dataset/domain, không phải model. Project áp dụng checkpoint pretrained, không train/fine-tune tập đích. Không suy ra benchmark hoàn toàn chưa xuất hiện trong training của model mới. Xem [paper](https://arxiv.org/abs/2104.08663).

## Research questions
RQ1: modern dense cải thiện so BM25/MiniLM trên dataset nào? RQ2: hybrid ổn định hơn dense khi đổi domain? RQ3: reranking tăng nDCG bao nhiêu và mất latency bao nhiêu? RQ4: tiến bộ modern retrieval giải quyết generalization đến mức nào trong phạm vi dataset đã đo?

## Baselines và modern systems
V1 giữ BM25 (rank_bm25 reference, tokenizer lowercase alphanumeric, k1=1.5/b=.75/epsilon=.25), MiniLM dense và BM25 → MiniLM CrossEncoder. Source nguyên bản trong legacy/v1; sáu JSON lịch sử results/ không bị ghi đè.
V2 thêm BGE-M3, Qwen3 Embedding 0.6B, hybrid BM25+dense RRF (k=60), BGE reranker và Qwen3 Reranker 0.6B. Qwen query có instruction; document không. Exact NumPy chunked và optional FAISS IndexFlatIP, chưa có ANN.

## Install: Windows PowerShell
Python 3.12 64-bit; tránh lệnh python toàn máy đang trỏ bản 32-bit.
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-core.txt
# CPU
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements-neural.txt
# Optional evaluator/index
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
# GPU: chọn wheel CUDA tương thích driver theo https://pytorch.org/get-started/locally/
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
GPU trong phần cứng không đồng nghĩa CUDA runtime khả dụng. CPU dùng FP32; CUDA auto BF16 khi hỗ trợ, còn lại FP16. Không bắt buộc FlashAttention trên Windows. Model weights tải qua TLS verified; REQUESTS_CA_BUNDLE hỗ trợ CA riêng, không tự bypass SSL.

## Install: Linux
```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements-core.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-neural.txt -r requirements-optional.txt
```
Chọn CUDA wheel theo driver nếu chạy GPU. requirements-lock.txt ghi dependency thực tế của phiên chạy này; không dùng lock Windows để suy ra GPU trên Linux.

## Quick start
Trong PowerShell dùng `.\.venv\Scripts\python.exe` thay `python` ở các command dưới. Trong Linux dùng python của venv.
```bash
python beir_lab.py run --dataset scifact --retriever bm25
python beir_lab.py run --dataset nfcorpus --retriever bm25
python beir_lab.py run --dataset scifact --retriever minilm
python beir_lab.py run --dataset scifact --retriever bge-m3 --device auto
python beir_lab.py run --config configs/hybrid.yaml --candidate-k 50
python beir_lab.py run --dataset scifact --retriever qwen3
python beir_lab.py run --dataset scifact --retriever hybrid --dense-model bge-m3 --reranker qwen3
```
Dataset hiện có chạy offline; thêm `--download` cho FiQA/ArguAna/TREC-COVID chưa có. Model tự dùng local cache nếu đã tải, chưa có cần mạng. CLI > YAML > defaults. Nested YAML hỗ trợ retriever/dense/bm25/fusion/reranker như configs/hybrid.yaml.

## Full benchmark và evaluation
```bash
python beir_lab.py benchmark --dataset scifact --all --time-budget-seconds 7200
python beir_lab.py evaluate --run runs/RUN_ID
python generate_report.py
python -m pytest -q
```
Matrix định nghĩa trong beir_lab.matrix(), includes candidate 20/50/100 và hybrid → Qwen. Benchmark từng experiment chạy process riêng, hết budget thì ghi NOT_RUN_RESOURCE_LIMIT. evaluate nhận run folder hoặc metrics.json trong run, không thể xác minh JSON lịch sử chỉ có aggregate metrics.
CLI cũ `run_bm25.py`, `run_dense.py`, `run_rerank.py` vẫn chạy, output chuyển sang runs riêng. collect_results.py và plot.py gọi report V2, giữ bảng lịch sử nguyên trạng.

## Results và artifacts
Lượt GPU tuần tự có giới hạn thời gian và theo dõi nhiệt độ:
```powershell
.\.venv\Scripts\python.exe run_gpu_session.py --budget-seconds 7200 --per-run-seconds 1200
```
Session ghi command, status và telemetry trong `reports/gpu_session_*`. Mỗi process dùng CUDA và hai CPU threads; model chạy lần lượt. Ngưỡng dừng 85°C là lựa chọn vận hành của supervisor. Lượt reranker BGE/Qwen dùng `--candidate-run runs/RUN_ID` để đọc candidate từ retriever đã VERIFIED; latency của lượt này chỉ đo reranker và được ghi `latency_scope=reranker_only`. Cần một experiment riêng không có `--candidate-run` để đo toàn pipeline. Session đầu dùng candidate-k=20; 50/100 vẫn là các experiment riêng.

Tiếp tục riêng Qwen với batch 1, một PyTorch CPU thread và điều tiết nhiệt:
```powershell
.\.venv\Scripts\python.exe run_gpu_session.py --dataset scifact --qwen-only --budget-seconds 7200 --per-run-seconds 5400 --temperature-stop 80 --cool-start-seconds 60 --cool-between-seconds 120 --start-temperature 55 --batch-rest-seconds 0.1
```
Để chạy NFCorpus với cùng cấu hình, thay `--dataset scifact` bằng `--dataset nfcorpus`. Mặc định dataset của supervisor vẫn là SciFact.

Chờ nhiệt <=55°C ổn định trước model mới, nghỉ ít nhất 60 giây ban đầu/120 giây giữa model; nghỉ 0,1 giây sau mỗi CUDA batch. Từ 73°C tạm nghỉ đến <=60°C, tối đa 600 giây; supervisor dừng ở 80°C. Đây là lựa chọn vận hành, không phải giới hạn nhiệt của nhà sản xuất. Timing/QPS bao gồm nghỉ và chờ nhiệt; config/protocol ghi rõ để không so trực tiếp với lượt không điều tiết tải. Giữ context 512 và warm-up 5/3 lượt đo; không đổi chất lượng đánh giá chỉ để tăng tốc.

Metric chính linear-gain nDCG@10, Recall@10/100, MRR@10, MAP@100. Gain mũ lưu riêng `legacy_exponential_ndcg@10`. Query judged thiếu ranking tính 0. Ranking TSV có IDs, score, rank. Mỗi run ghi config, metadata/checksum, metrics, environment, hardware, timing, log, status và validation.
VERIFIED nghĩa là ranking đã đọc lại, checksum/coverage hợp lệ, metric tính lại đúng; validation.json ghi pytrec_eval có được dùng hay không. Không đồng nghĩa tái hiện toàn bộ setup paper. Chỉ run thành công được biểu diễn bằng metric; failures/missing là N/A.

Cache embeddings/indexes/candidates có manifest; revision/checksum/dimension/thứ tự ID sai thì không reuse. Warm không dùng candidate cache, không cache query embedding. Cold log model load/download, index, corpus encoding/cache và first query pass riêng với warm. RAM=sampled RSS, VRAM=PyTorch peak nếu có; QPS serial. Xem docs/EXPERIMENT_DESIGN.md để diễn giải.

Report CSV/Markdown tại reports/, sáu hình tại plots/. Paper numbers trong references/beir_paper_results.csv, tách Paper result khỏi Our run và không ánh xạ tên reranker thành BM25. Kết quả cụ thể và failures xem reports/IMPLEMENTATION_REPORT.md.

## Tài liệu
Báo cáo chốt cuối cùng: [kết luận, metric có ý nghĩa thống kê và lựa chọn candidate-k](reports/paper_reproduction/final_conclusions_20261012_075023/comparison.md). Chọn **Qwen reranker top-50 cho SciFact**, **top-100 cho NFCorpus**; lý do chất lượng/chi phí và giới hạn phép đo được nêu rõ. `selected_config.json` ghi quyết định lựa chọn, không sửa config/kết quả của run cũ. Bản thống kê và các báo cáo trước được giữ nguyên.

Báo cáo cập nhật đủ nDCG@10, Recall@100, MRR@10 và MAP@100: [so sánh và thống kê đa metric](reports/paper_reproduction/multimetric_significance_20261012_073354/comparison.md). Ba metric bổ sung có Holm108 và sensitivity Holm144, giữ nguyên kết quả nDCG trước. Script `.\.venv-paper\Scripts\python.exe analyze_secondary_metrics.py` chỉ đọc ranking, tạo thư mục mới và lưu bằng chứng bảo toàn artifact/báo cáo cũ; không chạy model.

Bản nDCG trước đó: [kiểm định ý nghĩa thống kê nDCG theo query](reports/paper_reproduction/significance_comparison_20261012_070142/comparison.md): paired randomization + CI bootstrap 95%, Holm cho 36 phép so sánh. Có `significance.csv`, `per_query_metrics.csv`, protocol/seed và bằng chứng giữ nguyên artifact nguồn. Tái tạo bằng `.\.venv-paper\Scripts\python.exe analyze_significance.py`; không chạy model. Các bản báo cáo trước vẫn giữ nguyên.

Báo cáo so sánh hoàn chỉnh 7 hệ thống × 2 dataset: [ba baseline gốc, Qwen embedding và reranker top-20/50/100](reports/paper_reproduction/complete_comparison_20261012_064753/comparison.md). Có đủ năm metric, delta từng baseline, số paper tách riêng, chi phí và giới hạn phép đo, run ID và checksum. Script `report_verified_comparison.py` kiểm tra artifact theo cách chỉ đọc rồi tạo thư mục báo cáo mới; không chạy inference hoặc sửa kết quả cũ.

Phiên Qwen trên A30 ngày 12/10/2026 đã chạy đủ top-50/top-100 trên SciFact/NFCorpus, bốn run VERIFIED, không ghi đè top-20. Xem `reports/paper_reproduction/qwen_depths_20261012/comparison.md` và CSV cùng thư mục. Batch 1/BF16/context 512, nghỉ cố định 0; tạm dừng từ 73 đến 60°C, dừng cứng 80°C. Đây là quality-only, chưa đo warm latency/QPS. Tái tính báo cáo bằng `.\.venv-paper\Scripts\python.exe compare_qwen_depths.py`; launcher server dùng `.venv-paper/bin/python run_paper_neural_session.py --qwen` và luôn tạo run ID mới.

- docs/CODE_AUDIT.md: audit V1, correctness và phần giữ nguyên.
- docs/EXPERIMENT_DESIGN.md: thuật ngữ, protocol, metric, limitations.
- docs/REPORT_GUIDE.md và docs/SLIDE_GUIDE.md: đưa evidence vào Word/14 slide.
- BEIR_colab.ipynb: gọi cùng framework sau khi upload project đầy đủ.
