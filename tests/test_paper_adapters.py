import numpy as np
from lab.paper_adapters import cooled_encode, cooled_predict


class Gate:
    def __init__(self):
        self.before_count = self.after_count = 0

    def before(self):
        self.before_count += 1

    def after(self):
        self.after_count += 1


def test_cooled_encode_preserves_order_and_tensor_shape():
    import torch
    gate = Gate()
    seen = []
    def encode(items, **kwargs):
        assert len(items) == 1 and kwargs['batch_size'] == 1
        seen.extend(items)
        return torch.tensor([[items[0], items[0] + 1]])
    result = cooled_encode(encode, [3, 1, 2], gate, convert_to_tensor=True)
    assert seen == [3, 1, 2]
    assert result.tolist() == [[3, 4], [1, 2], [2, 3]]
    assert gate.before_count == gate.after_count == 3


def test_cooled_predict_keeps_all_candidates_and_empty_queries():
    gate = Gate()
    seen = []
    def predict(items, **kwargs):
        seen.extend(items)
        return np.array([len(items[0][1])], dtype=float)
    pairs = [('q', 'aaa'), ('q', 'b')]
    assert cooled_predict(predict, pairs, gate).tolist() == [3., 1.]
    assert cooled_predict(predict, [], gate).size == 0
    assert seen == pairs and gate.before_count == gate.after_count == 2
