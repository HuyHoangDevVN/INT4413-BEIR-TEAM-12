"""TREC-compatible linear-gain metrics; missing judged queries score zero."""
import math


def evaluate(results, qrels):
    totals = dict.fromkeys(['nDCG@10', 'Recall@10', 'Recall@100', 'MRR@10',
                          'MAP@100', 'legacy_exponential_ndcg@10'], 0.0)
    for qid, ranking in results.items():
        ids = [r['document_id'] if isinstance(r, dict) else r for r in ranking]
        if len(ids) != len(set(ids)):
            raise ValueError(f'Duplicate document in query {qid}')
        for row in ranking:
            if isinstance(row, dict) and not math.isfinite(float(row['score'])):
                raise ValueError('Non-finite ranking score')
    for qid, gold in qrels.items():
        ranked = results.get(qid, [])
        ids = [r['document_id'] if isinstance(r, dict) else r for r in ranked]
        positives = {d for d, rel in gold.items() if rel > 0}
        gains = [max(0, gold.get(d, 0)) for d in ids]
        ideal = sorted([max(0, r) for r in gold.values()], reverse=True)
        for exponential, key in [(False, 'nDCG@10'), (True, 'legacy_exponential_ndcg@10')]:
            def dcg(rels):
                return sum(((2 ** r - 1) if exponential else r) / math.log2(i + 2)
                           for i, r in enumerate(rels[:10]))
            denominator = dcg(ideal)
            totals[key] += dcg(gains) / denominator if denominator else 0
        for k in [10, 100]:
            totals[f'Recall@{k}'] += len(set(ids[:k]) & positives) / len(positives) if positives else 0
        totals['MRR@10'] += next((1 / i for i, d in enumerate(ids[:10], 1) if d in positives), 0)
        hits = 0
        ap = 0
        for i, d in enumerate(ids[:100], 1):
            if d in positives:
                hits += 1
                ap += hits / i
        totals['MAP@100'] += ap / len(positives) if positives else 0
    return {k: v / len(qrels) if qrels else 0.0 for k, v in totals.items()}
