"""
Score-based pruning for Bayesian network structure learning.
Uses bn_utils for _clamp_cpds, _refit_model, warn_if_bad_cpds.
"""

import pandas as pd
import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.metrics import structure_score

from bn_utils import _clamp_cpds, _refit_model, warn_if_bad_cpds

try:
    from bayesian_evaluation import build_pruning_row_extra
except ImportError:
    def build_pruning_row_extra(*args, **kwargs):
        raise ImportError("bayesian_evaluation.build_pruning_row_extra required")

try:
    from IPython.display import display
except ImportError:
    def display(x):
        print(x)


def evaluate_single_edge_deletions(
    current_model: DiscreteBayesianNetwork,
    data: pd.DataFrame,
    score_fn: str = "bic-d",
    verbose: bool = False,
):
    """Evaluate removing each edge using the given structure score. Fitted models are clamped."""
    base_edges = list(current_model.edges())
    nodes = list(current_model.nodes())
    candidates = []
    for edge in base_edges:
        try:
            new_edges = [e for e in base_edges if e != edge]
            pruned = DiscreteBayesianNetwork(new_edges)
            pruned.add_nodes_from(nodes)
            pruned.fit(data, estimator=MaximumLikelihoodEstimator)
            _clamp_cpds(pruned, min_prob=1e-10)
            score = structure_score(pruned, data, scoring_method=score_fn)
            candidates.append({"op": "remove", "edge": edge, "score": score, "model": pruned})
        except Exception as e:
            if verbose:
                print(f"[CANDIDATE FAIL] edge {edge}: {type(e).__name__}: {e}")
    return candidates


def score_pruning(
    true_model,
    pruned_model,
    data,
    evaluate_data,
    score_fn: str = "bic-d",
    max_steps=None,
    min_steps=None,
    evaluate_log_likelihood=None,
    evaluate_kl_divergence=None,
    evaluate_structural_error=None,
    target_var=None,
    interventions=None,
    evaluate_target_prediction_accuracy=None,
    evaluate_collider_preservation=None,
    evaluate_interventional_kl=None,
    evaluate_global_ace_difference=None,
):
    """
    Iteratively remove the edge whose removal gives the best structure score (e.g. BIC).
    Optional args can be omitted and read from global scope in a notebook.
    """
    g = globals()
    steps_max = max_steps if max_steps is not None else g.get("max_steps", 15)
    steps_min = min_steps if min_steps is not None else g.get("min_steps", 5)
    ll_fn = evaluate_log_likelihood or g.get("evaluate_log_likelihood")
    kl_fn = evaluate_kl_divergence or g.get("evaluate_kl_divergence")
    struct_fn = evaluate_structural_error or g.get("evaluate_structural_error")
    pred_fn = evaluate_target_prediction_accuracy or g.get("evaluate_target_prediction_accuracy")
    collider_fn = evaluate_collider_preservation or g.get("evaluate_collider_preservation")
    do_kl_fn = evaluate_interventional_kl or g.get("evaluate_interventional_kl")
    ace_fn = evaluate_global_ace_difference or g.get("evaluate_global_ace_difference")
    if ll_fn is None or kl_fn is None or struct_fn is None:
        raise ValueError(
            "evaluate_log_likelihood, evaluate_kl_divergence, evaluate_structural_error must be in scope or passed"
        )

    def _row_extra(true_m, learned_m, step_edges=None):
        return build_pruning_row_extra(
            true_m, learned_m, ll_fn, kl_fn, struct_fn, evaluate_data,
            step_edges=step_edges, target_var=target_var, pred_fn=pred_fn,
            collider_fn=collider_fn, interventions=interventions, do_kl_fn=do_kl_fn, ace_fn=ace_fn,
        )

    pruned_model = _refit_model(
        list(pruned_model.edges()),
        list(pruned_model.nodes()),
        data,
    )
    warn_if_bad_cpds(pruned_model)

    current_score = structure_score(pruned_model, data, scoring_method=score_fn)
    score_name = score_fn
    print(f"Baseline train {score_name}: {current_score:.3f}")

    baseline_extra = _row_extra(true_model, pruned_model, step_edges=0)
    history = [{
        "step": 0,
        "removed_edges": [],
        "num_edges": len(pruned_model.edges()),
        "score_name": score_fn,
        "score": current_score,
        **baseline_extra,
    }]

    print("\nStarting iterative pruning...")
    for step in range(1, steps_max + 1):
        candidates = evaluate_single_edge_deletions(pruned_model, data, score_fn=score_fn, verbose=False)
        if not candidates:
            print("No valid candidates left; stopping.")
            break
        best = max(candidates, key=lambda x: x["score"])
        previous_score = current_score
        current_score = best["score"]
        pruned_model = best["model"]
        print(f"\nSTEP {step}: removed edge {best['edge']}")
        print(f"  train {score_name} = {current_score:.6f} | delta = {current_score - previous_score:+.6f}")
        step_extra = _row_extra(true_model, pruned_model)
        history.append({
            "step": step,
            "removed_edges": best["edge"],
            "num_edges": len(pruned_model.edges()),
            "score_name": score_fn,
            "score": current_score,
            **step_extra,
        })
        if step >= steps_min and (current_score - previous_score) <= 0:
            print(f"No improvement after step {step}; stopping.")
            break

    print("\nPruning history:")
    display(pd.DataFrame(history))
    print("\nFinal model edges:")
    print(list(pruned_model.edges()))
    return pruned_model, history
