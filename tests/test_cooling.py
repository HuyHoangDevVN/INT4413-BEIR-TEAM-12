import pytest
from lab import cooling


def test_temperature_hysteresis_and_minimum_rest(monkeypatch):
    temperatures = iter([74, 68, 60])
    sleeps = []
    monkeypatch.setattr(cooling, 'gpu_temperature', lambda: next(temperatures))
    monkeypatch.setattr(cooling.time, 'sleep', sleeps.append)
    import torch
    synchronized = []
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: synchronized.append(True))
    gate = cooling.Cooling({'batch_rest_seconds': .1}, 'cuda')
    gate.before()
    gate.after()
    assert sleeps == [5, 5, .1]
    assert synchronized == [True]


def test_cpu_does_not_probe_gpu(monkeypatch):
    monkeypatch.setattr(cooling, 'gpu_temperature', lambda: pytest.fail('CPU must not probe GPU'))
    gate = cooling.Cooling({'batch_rest_seconds': .1}, 'cpu')
    gate.before()
    gate.after()


def test_thermal_control_without_fixed_rest(monkeypatch):
    temperatures = iter([73, 60])
    sleeps = []
    monkeypatch.setattr(cooling, 'gpu_temperature', lambda: next(temperatures))
    monkeypatch.setattr(cooling.time, 'sleep', sleeps.append)
    gate = cooling.Cooling({'batch_rest_seconds': 0, 'thermal_control': True}, 'cuda')
    gate.before()
    gate.after()
    assert sleeps == [5]


def test_thermal_limit_does_not_retry_as_oom():
    from lab.models import batched
    class Gate:
        def before(self):
            raise cooling.CoolingLimit('Cooling timeout')
    with pytest.raises(cooling.CoolingLimit, match='Cooling timeout'):
        batched([1], 8, lambda batch: pytest.fail('Must pause before inference'), 'cuda', Gate())


def test_qwen_final_logits_match_full_logits():
    import torch
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = Qwen3Config(vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=1,
                         num_attention_heads=2, num_key_value_heads=1, head_dim=8)
    model = Qwen3ForCausalLM(config).eval()
    ids = torch.tensor([[1, 2, 3]])
    with torch.inference_mode():
        full = model(input_ids=ids, use_cache=True).logits[:, -1, :]
        final_only = model(input_ids=ids, logits_to_keep=1, use_cache=False).logits[:, -1, :]
    torch.testing.assert_close(full, final_only, atol=1e-6, rtol=1e-5)
