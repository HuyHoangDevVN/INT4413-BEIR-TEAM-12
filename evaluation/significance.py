"""Paired query-level inference: sign-flip randomization, bootstrap, Holm."""
import numpy as np


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError('Invalid p-values')
    order = np.argsort(p, kind='stable')
    adjusted = np.empty_like(p)
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted


def paired_inference(first, second, seed=42, permutations=100000, bootstraps=20000):
    a, b = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    if a.ndim != 1 or a.shape != b.shape or len(a) == 0 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Finite equally sized paired vectors required')
    if permutations <= 0 or bootstraps <= 0:
        raise ValueError('Positive resample counts required')
    d = a-b
    delta = float(d.mean())
    rng = np.random.default_rng(seed)
    means = np.empty(bootstraps)
    for offset in range(0, bootstraps, 512):
        count = min(512, bootstraps-offset)
        means[offset:offset+count] = d[rng.integers(0, len(d), size=(count, len(d)))].mean(axis=1)
    ci = np.quantile(means, [.025, .975])
    nonzero = d[d != 0]
    threshold = abs(float(d.sum()))
    tolerance = 1e-12 * max(1., float(np.abs(d).sum()))
    exceedances = 0
    if len(nonzero) <= 16:
        total = 2**len(nonzero)
        for offset in range(0, total, 512):
            values = np.arange(offset, min(offset+512, total), dtype=np.uint64)
            signs = 2*((values[:, None] >> np.arange(len(nonzero), dtype=np.uint64)) & 1).astype(float)-1
            sums = (signs*nonzero).sum(axis=1)
            exceedances += int(np.count_nonzero(np.abs(sums) >= threshold-tolerance))
        pvalue = exceedances/total
        method = 'exact_sign_flip'
    else:
        total = permutations
        for offset in range(0, total, 512):
            count = min(512, total-offset)
            signs = 2*rng.integers(0, 2, size=(count, len(nonzero)), dtype=np.int8)-1
            sums = (signs*nonzero).sum(axis=1)
            exceedances += int(np.count_nonzero(np.abs(sums) >= threshold-tolerance))
        pvalue = (exceedances+1)/(total+1)
        method = 'monte_carlo_sign_flip_plus_one'
    return dict(mean_delta=delta, ci95_low=float(ci[0]), ci95_high=float(ci[1]), p_raw=pvalue,
                wins=int(np.count_nonzero(d > 1e-12)), losses=int(np.count_nonzero(d < -1e-12)),
                ties=int(np.count_nonzero(np.abs(d) <= 1e-12)), n_queries=len(d), seed=seed,
                permutation_method=method, permutation_samples=total, bootstrap_samples=bootstraps)
