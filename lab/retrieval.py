from collections import Counter, defaultdict
import heapq
import math
import re
import numpy as np


def tokenize(text):
    return re.findall(r'[a-z0-9]+', text.lower())


def topk(scores, ids, k):
    if k <= 0:
        raise ValueError('top-k must be positive')
    if not np.isfinite(scores).all():
        raise ValueError('Non-finite score')
    order = heapq.nsmallest(min(k, len(ids)), range(len(ids)), key=lambda j: (-float(scores[j]), j))
    return [{'document_id': ids[j], 'score': float(scores[j]), 'rank': i}
            for i, j in enumerate(order, 1)]


class BM25:
    """Postings version of rank_bm25.BM25Okapi (including epsilon IDF)."""
    def __init__(self, corpus, k1=1.5, b=0.75, epsilon=0.25):
        self.ids = list(corpus)
        self.k1, self.b = k1, b
        self.lengths = []
        self.postings = defaultdict(list)
        for j, (title, text) in enumerate(corpus.values()):
            terms = tokenize(f'{title} {text}')
            self.lengths.append(len(terms))
            for term, count in Counter(terms).items():
                self.postings[term].append((j, count))
        self.lengths = np.asarray(self.lengths)
        self.avgdl = float(self.lengths.mean())
        if not self.avgdl:
            raise ValueError('Corpus has no tokens')
        n = len(self.ids)
        self.idf = {t: math.log(n - len(p) + .5) - math.log(len(p) + .5) for t, p in self.postings.items()}
        floor = epsilon * sum(self.idf.values()) / len(self.idf)
        self.idf = {t: floor if v < 0 else v for t, v in self.idf.items()}

    def scores(self, query):
        scores = np.zeros(len(self.ids))
        for term in tokenize(query):
            postings = self.postings.get(term, [])
            if not postings:
                continue
            indices, counts = np.asarray(postings).T
            denominator = counts + self.k1 * (1 - self.b + self.b * self.lengths[indices] / self.avgdl)
            scores[indices] += self.idf[term] * (counts * (self.k1 + 1) / denominator)
        return scores

    def search(self, query, k):
        return topk(self.scores(query), self.ids, k)


def rrf(rankings, k=60, top_k=1000):
    if k < 0:
        raise ValueError('rrf-k must be nonnegative')
    scores = defaultdict(float)
    for ranking in rankings:
        seen = set()
        for rank, row in enumerate(ranking, 1):
            doc = row['document_id']
            if doc in seen:
                raise ValueError('Duplicate fusion candidate')
            seen.add(doc)
            scores[doc] += 1 / (k + rank)
    order = sorted(scores, key=lambda d: (-scores[d], d))[:top_k]
    return [{'document_id': d, 'score': scores[d], 'rank': i} for i, d in enumerate(order, 1)]


class VectorIndex:
    def __init__(self, embeddings, ids, backend='numpy', chunk_size=4096):
        self.embeddings, self.ids = embeddings, ids
        self.chunk_size = chunk_size
        self.backend = 'numpy_exact'
        self.index = None
        if backend == 'faiss':
            try:
                import faiss
                self.index = faiss.IndexFlatIP(embeddings.shape[1])
                self.index.add(np.asarray(embeddings, dtype=np.float32))
                self.backend = 'faiss_flat_exact'
            except ImportError:
                self.backend = 'numpy_exact_fallback_faiss_unavailable'

    def search(self, vector, k):
        if k <= 0 or self.chunk_size <= 0:
            raise ValueError('Search sizes must be positive')
        # FAISS computes all scores so ties have the same corpus-order contract.
        if self.index is not None:
            values, indices = self.index.search(np.asarray(vector, dtype=np.float32).reshape(1, -1), len(self.ids))
            scores = np.empty(len(self.ids), dtype=np.float32)
            scores[indices[0]] = values[0]
            return topk(scores, self.ids, k)
        best = []
        for start in range(0, len(self.ids), self.chunk_size):
            scores = self.embeddings[start:start + self.chunk_size] @ vector
            if not np.isfinite(scores).all():
                raise ValueError('Non-finite similarity')
            for offset, score in enumerate(scores):
                j = start + offset
                item = (float(score), -j)
                if len(best) < min(k, len(self.ids)):
                    heapq.heappush(best, item)
                elif item > best[0]:
                    heapq.heapreplace(best, item)
        return [{'document_id': self.ids[-neg_j], 'score': score, 'rank': i}
                for i, (score, neg_j) in enumerate(sorted(best, reverse=True), 1)]
