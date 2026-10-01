import numpy as np
import pytest
from evaluation.significance import paired_inference, holm


def test_exact_two_sided_and_zero_pairs():
    result = paired_inference([1, 1, 1, 0], [0, 0, 0, 0], bootstraps=100)
    assert result['p_raw'] == .25  # all-positive and all-negative among eight assignments
    assert result['mean_delta'] == .75
    assert (result['wins'], result['losses'], result['ties']) == (3, 0, 1)
    same = paired_inference([.2, .5], [.2, .5], bootstraps=100)
    assert same['p_raw'] == 1 and same['ci95_low'] == same['ci95_high'] == 0


def test_holm_reference_and_order():
    assert np.allclose(holm([.01, .04, .03, .002]), [.03, .06, .06, .008])


def test_monte_carlo_reproducibility_and_pair_direction():
    a = np.linspace(0, 1, 20)
    r = paired_inference(a, np.zeros(20), permutations=1000, bootstraps=200)
    assert r == paired_inference(a, np.zeros(20), permutations=1000, bootstraps=200)
    assert r['permutation_method'] == 'monte_carlo_sign_flip_plus_one'
    assert r['p_raw'] > 0
    opposite = paired_inference(np.zeros(20), a, permutations=1000, bootstraps=200)
    assert opposite['p_raw'] == r['p_raw']
    assert opposite['mean_delta'] == -r['mean_delta']


def test_invalid_pairing():
    with pytest.raises(ValueError):
        paired_inference([1, 2], [1])
    with pytest.raises(ValueError):
        paired_inference([float('nan')], [0])
