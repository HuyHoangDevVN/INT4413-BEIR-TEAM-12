# Thiết kế thực nghiệm BEIR V2

## Bài toán và thuật ngữ
Corpus là tập document; query là nhu cầu tìm kiếm; qrels là các đánh giá relevance do dataset cung cấp. BEIR là benchmark gồm nhiều tác vụ/domain, không phải một model. Xem [paper BEIR 2021](https://arxiv.org/abs/2104.08663) và [implementation evaluator](https://github.com/beir-cellar/beir/blob/main/beir/retrieval/evaluation.py).

Zero-shot trong project nghĩa là áp dụng checkpoint pretrained, không train/fine-tune trên tập đích. Điều này không chứng minh tập đích chưa từng xuất hiện trong training của checkpoint. Generalization là khả năng giữ chất lượng khi dữ liệu/tác vụ thay đổi; robustness được kiểm tra qua nhiều dataset, mức giảm tệ nhất và số dataset thắng/thua baseline. Hai dataset chỉ cho bằng chứng giới hạn, không đủ kết luận cho toàn BEIR.

Lexical retrieval đối chiếu từ vựng. BM25 thưởng term hiếm và điều chỉnh term frequency/độ dài document; tokenizer hiện tại là lowercase `[a-z0-9]+`, không stem. Sparse retrieval dùng vector chủ yếu bằng 0, có thể là trọng số từ cổ điển hoặc trọng số learned; V2 chưa triển khai learned sparse. Dense retrieval mã hóa text thành vector dày, có thể tìm tương đồng dù thiếu từ trùng. Bi-encoder encode query/document độc lập nên document vector có thể cache. Cross-encoder đọc cặp query/document cùng nhau, đắt hơn nhưng thấy tương tác từ ngữ.

Hybrid kết hợp lexical và dense. RRF tính `sum(1/(k+rank))`, rank từ 1, mặc định k=60. Nó không cộng trực tiếp hai thang score khác nhau. Reranking chỉ đổi thứ tự candidates đã truy xuất; không thể phục hồi document ngoài candidate set. Với cùng 100 document, Recall@100 phải bất biến sau rerank. Với candidate-k=20/50, Recall@100 bị giới hạn bởi tập nhỏ hơn; so sánh đúng với retriever đã cắt cùng K.

Exact vector search duyệt đủ vectors. NumPy chunked và FAISS IndexFlatIP là exact; V2 không có ANN. ANN (approximate nearest neighbours) đánh đổi độ chính xác tìm kiếm để giảm chi phí, cần đánh giá recall của index riêng nếu bổ sung.

## Metric và quy ước
- nDCG@10: DCG=`sum(rel/log2(rank+1))`, chia DCG lý tưởng của qrels. Gain tuyến tính để đối chiếu TREC. Gain mũ được lưu riêng dưới `legacy_exponential_ndcg@10`.
- Recall@K: số relevant retrieved trong top-K chia tổng document có relevance >0 của query.
- MRR@10: nghịch đảo rank relevant đầu tiên trong top-10; không có thì 0.
- MAP@100: tổng precision tại rank relevant trong top-100 chia tổng relevant của query, rồi macro-average; không chia bằng min(100,num_relevant).
- Mọi query có qrels thuộc split đều được tính. Query thiếu kết quả là 0. Unjudged document có relevance 0; qrels không hoàn chỉnh là một hạn chế.
- Ranking không duplicate, score hữu hạn, rank liên tục. BM25/vector ties giữ corpus order; RRF ties dùng document ID; reranker ties giữ candidate order.
- Mặc định bỏ document có ID giống query theo BEIR, đặc biệt quan trọng với ArguAna. Có thể tắt bằng --no-ignore-identical-ids; đây là experiment khác. Evaluator TREC nhận ranking sau bước này. SciFact/NFCorpus local không có query/document ID overlap.

## RQ và matrix
RQ1 so sánh BM25/MiniLM với BGE-M3 và Qwen3 trên cùng dataset/split/evaluator. RQ2 so dense với hybrid theo từng dataset và thống kê số lần thắng, worst-domain delta. RQ3 đo delta reranker với retriever đã cắt cùng candidate K=20/50/100, kèm tỷ lệ latency. RQ4 trình bày mức cải thiện quan sát được, không tuyên bố modern model giải quyết hoàn toàn vấn đề BEIR.

Matrix trong `beir_lab.matrix()` gồm V1, BGE dense/hybrid, các nguồn candidate → BGE reranker, Qwen dense → Qwen reranker và hybrid → Qwen reranker. Không tune theo test qrels. Mặc định MiniLM giữ max_length của checkpoint (256); BGE/Qwen dense và CE=512 (truncation được log), seed=42, k1=1.5, b=.75, epsilon=.25. Có thể tăng context bằng config; coi đó là experiment mới.

## Kiểm định thống kê từ ranking có sẵn

`analyze_significance.py` chỉ đọc ranking và dùng CPU. Metric chính là nDCG@10 gain tuyến tính, đơn vị ghép cặp là query có qrels cùng dataset. Hai dataset tạo 36 phép so sánh (24 Qwen với ba baseline gốc, 6 giữa candidate-k, 6 reranker với embedding); hiệu chỉnh Holm trên toàn family, alpha 0,05, kiểm định hai phía. Dùng sign-flip randomization 100.000 mẫu với plus-one p-value; exact nếu ≤16 cặp khác 0. CI paired percentile bootstrap 95% dùng 20.000 mẫu, pointwise chưa hiệu chỉnh. Seed cố định được lưu trong statistical_method.json.

Kết luận tăng có ý nghĩa cần mean delta >0 và p Holm <0,05. CI pointwise không thay thế quyết định nhiều phép thử. Chưa kiểm định các metric phụ; không bác bỏ H0 không chứng minh tương đương. Đây là phân tích hậu nghiệm, phụ thuộc giả định query có thể lấy mẫu/hoán đổi phù hợp; không bao quát model training seed, domain chưa đo, thiếu qrels hoặc contamination. Không chạy lại model và không sửa artifact nguồn; báo cáo cập nhật nằm trong thư mục mới.

Kiểm định bổ sung qua `analyze_secondary_metrics.py` dùng cùng 36 so sánh cho Recall@100, MRR@10, MAP@100: Holm chung 108 phép bổ sung, kèm sensitivity Holm144 gồm cả nDCG. Các quyết định nDCG family36 cũ được bảo toàn. Phép Monte Carlo có p Holm pilot trong (0,025; 0,075) được tính lại với 1 triệu mẫu và seed độc lập, rồi hiệu chỉnh lại toàn family; protocol ghi rõ refinement hậu nghiệm. Recall@10/gain mũ chưa kiểm định.

## Đo chi phí
Protocol có `batch_rest_seconds>0` là lượt điều tiết tải: CUDA synchronize rồi nghỉ sau từng batch, với hysteresis nhiệt theo `cooling_pause_c`/`cooling_resume_c`. Latency và QPS bao gồm thời gian nghỉ/chờ nguội, không phải latency GPU chạy liên tục. Chỉ so chi phí với protocol nghỉ giống nhau; các lượt không có nghỉ vẫn giữ nguyên. Các ngưỡng là lựa chọn vận hành được log, không phải chẩn đoán lỗi Windows hay cam kết nhiệt an toàn tuyệt đối.

Cold trong một process mới gồm dataset load/download, model load/download, embedding (hoặc cache hit), index build/load và first query pass. Các run mới tách Hub download/cache resolution khỏi model construction; các run đầu dùng constructor trực tiếp có trường gộp model_load_download_seconds, phải đọc schema timing của từng run. Warm: model/index đã load, corpus embeddings đã sẵn; 5 query warmup, 3 lượt toàn bộ queries, đo query encoding + retrieve + fusion + rerank. Không dùng candidate cache khi đo latency; embedding query không cache. CUDA synchronize trước/sau phép đo.

RAM peak lấy RSS process mỗi 10ms; có thể bỏ lỡ spike ngắn. VRAM allocated/reserved peak do PyTorch, không phải tổng GPU memory của mọi process. QPS là throughput serial, không đại diện concurrent serving. Index size là serialized index hoặc resident NumPy vectors; log backend để phân biệt. Các giá trị không khả dụng để null.

## Tái lập và trạng thái
Chế độ `--candidate-run` kiểm tra lại artifact nguồn và dùng ranking cố định của retriever cùng dataset. Chỉ reranker được load; quality là kết quả xếp lại candidate, warm latency chỉ là reranker (`latency_scope=reranker_only`). Không so tỷ lệ latency của chế độ này với pipeline end-to-end. Supervisor `run_gpu_session.py` giới hạn tổng thời gian, từng experiment và nhiệt độ; dừng cây process thuộc experiment khi vượt giới hạn. Đây là giới hạn vận hành, không chẩn đoán hay sửa lỗi HYPERVISOR_ERROR.

Mỗi run ghi config resolved, ranking TSV, checksums, metadata, environment/hardware/timing/log/status. Revision model resolve từ checkpoint. `RUN_SUCCESS` sau hoàn tất; `VERIFIED` sau đọc lại ranking, checksum/coverage và tính metric lại. Nếu có pytrec_eval, đối chiếu năm metric; không có thì validation ghi unavailable. `VERIFIED` không đồng nghĩa tái hiện đúng tất cả setup của paper.

`NOT_RUN` chưa thực thi; `NOT_RUN_RESOURCE_LIMIT` thiếu tài nguyên hoặc hết ngân sách wall-clock; `NOT_RUN_MODEL_DOWNLOAD_FAILED` thất bại trong tải/load model; `RUN_FAILED` lỗi còn lại. Không aggregate metric của lượt thất bại. JSON lịch sử không có ranking/environment nên giữ riêng, không nâng trạng thái.

## Nguồn model và hạn chế
[BGE-M3](https://huggingface.co/BAAI/bge-m3), [BGE reranker](https://huggingface.co/BAAI/bge-reranker-v2-m3), [Qwen embedding](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B), [Qwen reranker](https://huggingface.co/Qwen/Qwen3-Reranker-0.6B). Adapter Qwen query dùng instruction; document không thêm instruction. Qwen reranker dùng causal LM template và probability yes/no theo model card. Các adapter chưa thực chạy phải được ghi là implementation/test-fixture evidence.

Đợt đầu chạy đầy đủ BM25 trên hai dataset; neural có tối đa 2 giờ. Download failure, CPU inference chậm hoặc budget hết đều được ghi thật, không sinh metric thay thế. Rerank candidate sizes, model revision/context và hardware là biến thí nghiệm cần giữ khi diễn giải quality-cost.
