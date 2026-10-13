import time
import numpy as np
from .cache import EmbeddingCache
from .data import ROOT
from .cooling import Cooling, CoolingLimit

DENSE_MODELS = {'minilm': 'sentence-transformers/all-MiniLM-L6-v2',
                'bge-m3': 'BAAI/bge-m3', 'qwen3': 'Qwen/Qwen3-Embedding-0.6B'}
RERANK_MODELS = {'minilm': 'cross-encoder/ms-marco-MiniLM-L-6-v2',
                 'bge-reranker-v2-m3': 'BAAI/bge-reranker-v2-m3',
                 'qwen3': 'Qwen/Qwen3-Reranker-0.6B'}
TASK = 'Given a web search query, retrieve relevant passages that answer the query'


def checkpoint(name, revision=None):
    """Time Hub transfer/cache resolution separately from model construction."""
    from pathlib import Path
    from huggingface_hub import snapshot_download, HfApi
    if Path(name).is_dir():
        from .data import digest
        from .cache import file_digest
        files = [(str(p.relative_to(name)), file_digest(p))
                 for p in sorted(Path(name).rglob('*')) if p.is_file()]
        return name, 'local-' + digest(files), 0.0
    start = time.perf_counter()
    info = HfApi().model_info(name, revision=revision)
    weights = '*.safetensors' if any('/' not in f.rfilename and f.rfilename.endswith('.safetensors') for f in info.siblings) else '*.bin'
    path = snapshot_download(name, revision=info.sha,
                             allow_patterns=['*.json', '*.txt', weights, '*.model', 'tokenizer*'],
                             ignore_patterns=['onnx/*', 'openvino/*', '*.onnx', '*.onnx_data'], max_workers=1)
    return path, Path(path).name, time.perf_counter() - start


def resolve_device(requested):
    import torch
    if requested == 'cuda' and not torch.cuda.is_available():
        raise MemoryError('CUDA requested but unavailable')
    return 'cuda' if requested != 'cpu' and torch.cuda.is_available() else 'cpu'


def dtype_for(device):
    import torch
    return torch.float32 if device == 'cpu' else (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16)


def batched(items, batch_size, fn, device, cooling=None, progress_interval=0):
    import torch
    outputs = []
    start = 0
    next_progress = progress_interval
    while start < len(items):
        try:
            if cooling:
                cooling.before()
            output = fn(items[start:start + batch_size])
            if cooling:
                cooling.after()
            outputs.extend(output)
            start += min(batch_size, len(items) - start)
            if progress_interval and len(items) >= progress_interval and (start >= next_progress or start == len(items)):
                print(f'Encoded {start}/{len(items)}', flush=True)
                next_progress = (start // progress_interval + 1) * progress_interval
        except (torch.OutOfMemoryError, MemoryError) as exc:
            if isinstance(exc, CoolingLimit):
                raise
            if batch_size <= 1:
                raise MemoryError('OOM at batch size 1') from exc
            batch_size = max(1, batch_size // 2)
            print(f'OOM: retry current batch with size {batch_size}', flush=True)
            if device == 'cuda':
                torch.cuda.empty_cache()
    return np.asarray(outputs)


class Dense:
    def __init__(self, config, device):
        from sentence_transformers import SentenceTransformer
        self.config, self.device = config, device
        self.cooling = Cooling(config, device)
        self.name = DENSE_MODELS.get(config['dense_model'], config['dense_model'])
        path, revision, self.download_time = checkpoint(self.name, config.get('model_revision'))
        start = time.perf_counter()
        self.model = SentenceTransformer(path, device=device,
                                         model_kwargs={'torch_dtype': dtype_for(device)})
        self.load_time = time.perf_counter() - start
        self.config = dict(config)
        if config['max_length'] is None:
            self.config['max_length'] = self.model.max_seq_length if 'all-MiniLM' in self.name else 512
        self.model.max_seq_length = self.config['max_length']
        self.revision = revision
        if not self.revision:
            raise ValueError('Could not resolve immutable model revision; cache disabled by refusing run')
        self.prompt = (self.model.prompts.get('query') or f'Instruct: {TASK}\nQuery:') if 'Qwen3' in self.name else ''

    def encode(self, texts, query=False):
        prompt = self.prompt if query else ''
        matrix = batched(texts, self.config['batch_size'],
                         lambda batch: self.model.encode(batch, prompt=prompt, convert_to_numpy=True,
                                                        normalize_embeddings=True, show_progress_bar=False), self.device, self.cooling,
                         self.config.get('progress_interval', 0))
        if not np.isfinite(matrix).all() or np.any(np.linalg.norm(matrix, axis=1) == 0):
            raise ValueError('Invalid embedding')
        return np.asarray(matrix, dtype=np.float32)

    def corpus(self, corpus, metadata):
        ids = list(corpus)
        folder = ROOT / 'cache' / 'embeddings' / self.name.replace('/', '--') / metadata['dataset']
        cache = EmbeddingCache(folder)
        manifest = {**metadata, 'model': self.name, 'revision': self.revision, 'dtype': str(dtype_for(self.device)),
                    'max_length': self.config['max_length'], 'normalization': 'l2', 'text': 'title + space + text',
                    'dimension': self.model.get_sentence_embedding_dimension()}
        matrix = cache.load(manifest, ids)
        start = time.perf_counter()
        hit = matrix is not None
        if not hit:
            matrix = self.encode([f'{t} {x}'.strip() for t, x in corpus.values()])
            cache.save(manifest, ids, matrix)
        return matrix, ids, folder, time.perf_counter() - start, hit, manifest


class Reranker:
    def __init__(self, config, device):
        import torch
        self.config, self.device = config, device
        self.cooling = Cooling(config, device)
        self.name = RERANK_MODELS.get(config['reranker'], config['reranker'])
        path, self.revision, self.download_time = checkpoint(self.name, config.get('reranker_revision'))
        start = time.perf_counter()
        if 'Qwen3' in self.name:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left')
            self.model = AutoModelForCausalLM.from_pretrained(path,
                                                            torch_dtype=dtype_for(device)).to(device).eval()
            prefix = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
            suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
            self.prefix = self.tokenizer.encode(prefix, add_special_tokens=False)
            self.suffix = self.tokenizer.encode(suffix, add_special_tokens=False)
            self.yes = self.tokenizer.convert_tokens_to_ids('yes')
            self.no = self.tokenizer.convert_tokens_to_ids('no')
        else:
            from sentence_transformers import CrossEncoder
            self.model = CrossEncoder(path, device=device,
                                      max_length=config['reranker_max_length'], model_kwargs={'torch_dtype': dtype_for(device)})
        self.load_time = time.perf_counter() - start

    def predict(self, pairs):
        if 'Qwen3' not in self.name:
            return self.model.predict(pairs, batch_size=len(pairs), show_progress_bar=False)
        import torch
        texts = [f'<Instruct>: {TASK}\n<Query>: {q}\n<Document>: {d}' for q, d in pairs]
        limit = self.config['reranker_max_length'] - len(self.prefix) - len(self.suffix)
        if limit <= 0:
            raise ValueError('reranker_max_length too small for Qwen template')
        tokens = self.tokenizer(texts, padding=False, truncation=True, max_length=limit, add_special_tokens=False)
        ids = [self.prefix + t + self.suffix for t in tokens['input_ids']]
        inputs = self.tokenizer.pad({'input_ids': ids}, padding=True, return_tensors='pt').to(self.device)
        with torch.inference_mode():
            # Only the final yes/no logits are scored; generation KV cache is unnecessary.
            logits = self.model(**inputs, logits_to_keep=1, use_cache=False).logits[:, -1, :]
            score = torch.softmax(logits[:, [self.no, self.yes]].float(), dim=1)[:, 1]
        return score.cpu().numpy()

    def rerank(self, query, candidates, corpus):
        pairs = [(query, f'{corpus[r["document_id"]][0]} {corpus[r["document_id"]][1]}'.strip()) for r in candidates]
        scores = batched(pairs, self.config['reranker_batch_size'], self.predict, self.device, self.cooling)
        order = sorted(range(len(candidates)), key=lambda j: (-float(scores[j]), j))
        return [{'document_id': candidates[j]['document_id'], 'score': float(scores[j]), 'rank': i}
                for i, j in enumerate(order, 1)]
