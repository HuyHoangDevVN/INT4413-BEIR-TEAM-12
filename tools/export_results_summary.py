"""Publish compact summaries from an existing evidence package; never run retrieval."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--handoff', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'docs/results')
    args = parser.parse_args()
    source = args.handoff.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / 'docs'):
        parser.error('Summary output must stay under docs/')
    manifest = json.loads((source / 'runs_manifest.json').read_text(encoding='utf-8'))
    output.mkdir(parents=True, exist_ok=False)
    provenance = []
    for relative in ['collection_summary.json', 'tables/quality.csv', 'tables/comparisons.csv',
                     'tables/statistical_summary.csv', 'tables/efficiency.csv', 'candidate_checks.json',
                     'evaluation_protocol.json', 'report_selection.json']:
        target = output / Path(relative).name
        shutil.copyfile(source / relative, target)
        provenance.append(dict(file=target.name,source=relative,
                               sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
    (output / 'provenance.json').write_text(json.dumps(dict(
        evidence_collected_utc=manifest['collected_utc'],source_files=provenance,
        note='Read-only summaries exported from the local evidence package; ranking archive not distributed in Git'),
        ensure_ascii=False, indent=2), encoding='utf-8')
    by_id = {r['run_id']: r for r in manifest['runs']}
    rows = [by_id[rid] for rid in manifest['primary_run_ids']]
    assert all(r['verification_status'] == 'metrics_recomputed_match' for r in rows)
    text = '# Kết quả baseline và Qwen đã kiểm chứng\n\n'
    text += ('Snapshot: ' + manifest['collected_utc'] + '. Không đại diện tiến độ live của server.\n\n'
             'Metric dưới đây được tính lại từ ranking đã lưu và đối chiếu pytrec_eval trong gói bằng chứng local. '
             'Ranking/qrels đầy đủ không nằm trong Git; `provenance.json` ghi nguồn và checksum của các bảng. '
             'SHA-256 chứng minh tính toàn vẹn, không tự chứng minh thí nghiệm đúng.\n\n')
    text += '| Dataset | Pipeline | Run ID | nDCG@10 | Recall@100 | MRR@10 | MAP@100 |\n'
    text += '| --- | --- | --- | ---: | ---: | ---: | ---: |\n'
    for row in rows:
        values = row['recomputed_metrics']
        text += '| ' + ' | '.join([row['dataset'],row['system'],row['run_id']] +
                    [f'{values[m]:.6f}' for m in ['nDCG@10','Recall@100','MRR@10','MAP@100']]) + ' |\n'
    text += ('\n## Điều kiện diễn giải\n\n'
             '- `quality.csv` giữ mọi hồ sơ, kể cả partial/failed/summary-only; không lấy điểm test cao nhất làm cấu hình chính.\n'
             '- `comparisons.csv` chứa delta hợp lệ và run ID đối chứng; điểm paper nằm riêng trong `references/`.\n'
             '- `statistical_summary.csv` là kiểm định đã lưu: nDCG Holm36, metric bổ sung Holm108 và sensitivity Holm144.\n'
             '- Top-50 SciFact/top-100 NFCorpus là lựa chọn post-hoc, không phải tuning trên dev độc lập.\n'
             '- Top-20/50 có Recall@100 trên output đã cắt; reranker100 giữ Recall@100 của cùng 100 candidates.\n'
             '- Laptop/A30/CPU và scope timing khác nhau: không kết luận speedup trực tiếp từ `efficiency.csv`.\n'
             '- Hai dataset có đủ matrix baseline/Qwen. FiQA/ArguAna/TREC-COVID trong snapshot chỉ có một phần hệ thống.\n'
             '- Không chứng minh strict zero-shot hoặc cải thiện trên toàn BEIR.\n')
    (output / 'README.md').write_text(text, encoding='utf-8')
    print('Exported compact verified summaries:', output)


if __name__ == '__main__':
    main()
