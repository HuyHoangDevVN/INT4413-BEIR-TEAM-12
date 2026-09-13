"""Batch-one compatibility shims; upstream text preparation and scoring remain intact."""
import numpy as np


def cooled_encode(encode, items, gate, **kwargs):
    import torch
    kwargs.update(batch_size=1, show_progress_bar=False)
    if not items:
        return encode(items, **kwargs)
    outputs = []
    for i, item in enumerate(items):
        gate.before()
        outputs.append(encode([item], **kwargs))
        gate.after()
        if (i + 1) % 500 == 0:
            print(f'Encoded {i + 1}/{len(items)}', flush=True)
    return torch.cat(outputs) if isinstance(outputs[0], torch.Tensor) else np.concatenate(outputs)


def cooled_predict(predict, sentences, gate):
    outputs = []
    for item in sentences:
        gate.before()
        outputs.extend(predict([item], batch_size=1, show_progress_bar=False))
        gate.after()
    return np.asarray(outputs)
