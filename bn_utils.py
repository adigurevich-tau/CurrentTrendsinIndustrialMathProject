"""
Shared utilities for Bayesian network pruning: model refit, CPD clamping, and validation.
Used by wavelet, score, and CSI pruning modules.
"""

import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator


def clamp_cpd_values(model, min_prob=1e-3):
    """Clamp CPD values to min_prob, renormalize, and optionally print when clamping is heavy."""
    for cpd in model.get_cpds():
        values = cpd.values.copy()
        n_clamped = np.sum(values < min_prob)
        if n_clamped > 0:
            print(f"  {cpd.variable}: clamped {n_clamped}/{values.size} entries")
        values = np.clip(values, min_prob, None)
        values /= values.sum(axis=0, keepdims=True)
        cpd.values = values
        frac = n_clamped / values.size
        if frac > 0.05:
            print(f"[CLAMP HEAVY] {cpd.variable}: clamped {frac:.1%} of entries at {min_prob}")
    return model


def add_gaussian_index_noise(df, eps=0.3, seed=0):
    """Add index-based noise: with probability eps, shift each value to a neighboring state."""
    rng = np.random.default_rng(seed)
    noisy = df.copy()
    for col in noisy.columns:
        states = np.sort(noisy[col].unique())
        k = len(states)
        idx = {s: i for i, s in enumerate(states)}
        mask = rng.random(len(noisy)) < eps
        for i in noisy[mask].index:
            cur = idx[noisy.at[i, col]]
            shift = rng.choice([-1, 1])
            noisy.at[i, col] = states[(cur + shift) % k]
    return noisy


def _clamp_cpds(model, min_prob=1e-10):
    """Clamp CPD values to min_prob and renormalize so log-likelihood never sees log(0)."""
    for cpd in model.get_cpds():
        vals = np.clip(cpd.values, min_prob, None)
        vals = vals / vals.sum(axis=0, keepdims=True)
        cpd.values = vals


def _refit_model(edges, nodes, data):
    """Build and fit a DiscreteBayesianNetwork with MLE, then clamp CPDs to avoid log(0)."""
    m = DiscreteBayesianNetwork(edges)
    m.add_nodes_from(nodes)
    m.fit(data, estimator=MaximumLikelihoodEstimator)
    _clamp_cpds(m, min_prob=1e-10)
    return m


def warn_if_bad_cpds(model, atol=1e-6):
    """Print a warning if any CPD has non-finite values or columns not summing to 1."""
    for cpd in model.get_cpds():
        vals = np.asarray(cpd.values, dtype=float)
        if not np.isfinite(vals).all():
            print(f"[BAD CPD] {cpd.variable}: has NaN/Inf values")
            return
        vcard = cpd.variable_card
        mat = vals.reshape(vcard, -1)
        col_sums = mat.sum(axis=0)
        if not np.allclose(col_sums, 1.0, atol=atol):
            worst = float(np.max(np.abs(col_sums - 1.0)))
            print(f"[BAD CPD] {cpd.variable}: columns not normalized (max |sum-1| = {worst:.2e})")
            return
