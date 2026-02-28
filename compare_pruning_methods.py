"""
Compare Wavelet, BIC, AIC, BDs, and CSI pruning methods (predictive + causal metrics).

In a notebook, add the folder that contains bayesian_evaluation.py to sys.path, then import:

  import sys
  sys.path.insert(0, '/content')   # Colab: folder where you uploaded the .py files
  import bayesian_evaluation
  from compare_pruning_methods import run_and_print_comparison, plot_pruning_progress
  df = run_and_print_comparison(..., evaluation_module=bayesian_evaluation)
  plot_pruning_progress({"Wavelet": wavelet_history, "BIC": BIC_history, ...})
"""

import os
import sys
import numpy as np
import pandas as pd
from pgmpy.models import DiscreteBayesianNetwork


def _get_evaluation_module(evaluation_module=None):
    """Use provided module or try to import bayesian_evaluation."""
    if evaluation_module is not None:
        return evaluation_module
    import importlib.util
    try:
        import bayesian_evaluation as ev
        return ev
    except ModuleNotFoundError:
        pass
    cwd = os.getcwd()
    if cwd not in sys.path:
        sys.path.insert(0, cwd)
    try:
        import bayesian_evaluation as ev
        return ev
    except ModuleNotFoundError:
        pass
    try:
        this_dir = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(this_dir, "bayesian_evaluation.py")
        if os.path.isfile(path):
            spec = importlib.util.spec_from_file_location("bayesian_evaluation", path)
            ev = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(ev)
            return ev
    except Exception:
        pass
    raise ModuleNotFoundError(
        "Could not import bayesian_evaluation.\n"
        "1. Put bayesian_evaluation.py in the same folder as compare_pruning_methods.py.\n"
        "2. In the notebook, add that folder to sys.path, then import and pass it:\n"
        "   import sys\n"
        "   sys.path.insert(0, '/content')  # or your folder path\n"
        "   import bayesian_evaluation\n"
        "   run_and_print_comparison(..., evaluation_module=bayesian_evaluation)"
    )


def run_comparison_table(
    true_model: DiscreteBayesianNetwork,
    models: dict,
    data: pd.DataFrame,
    evaluate_data: pd.DataFrame,
    target_var: str = None,
    interventions: list = None,
    n_kl: int = 500,
    n_do: int = 300,
    verbose: bool = True,
    evaluation_module=None,
) -> pd.DataFrame:
    """
    Build a single DataFrame with one row per pruning method and columns:
    - Predictive: neg_ll (generalization), kl_div (to true), pred_accuracy (target)
    - Causal: structural_error, collider_recall, collider_precision, interventional_kl_mean
    If evaluation_module is None, tries to import bayesian_evaluation; pass it in a notebook if import fails.
    """
    ev = _get_evaluation_module(evaluation_module)

    rows = []
    for name, learned in models.items():
        if verbose:
            print(f"  {name}...")
        r = {"method": name}
        r["neg_ll"] = ev.evaluate_log_likelihood(learned, evaluate_data)
        r["kl_div"] = ev.evaluate_kl_divergence(true_model, learned, n_samples=n_kl, verbose=False)
        r["structural_error"] = ev.evaluate_structural_error(true_model, learned)
        if target_var and target_var in learned.nodes():
            r["pred_accuracy"] = ev.evaluate_target_prediction_accuracy(
                learned, evaluate_data, target_var
            )
        else:
            r["pred_accuracy"] = np.nan
        coll = ev.evaluate_collider_preservation(true_model, learned)
        r["collider_recall"] = coll["recall"]
        r["collider_precision"] = coll["precision"]
        if hasattr(ev, "evaluate_global_ace_difference"):
            try:
                r["global_ace_diff"] = ev.evaluate_global_ace_difference(true_model, learned)
            except Exception:
                r["global_ace_diff"] = np.nan
        else:
            r["global_ace_diff"] = np.nan
        if interventions:
            kls = []
            for do_dict in interventions:
                kl = ev.evaluate_interventional_kl(
                    true_model, learned, do_dict, n_samples=n_do, verbose=False
                )
                if not np.isnan(kl):
                    kls.append(kl)
            r["interventional_kl_mean"] = float(np.mean(kls)) if kls else np.nan
        else:
            r["interventional_kl_mean"] = np.nan
        rows.append(r)

    return pd.DataFrame(rows)


def comparison_table_from_histories(histories_dict, method_order=None):
    """
    Build the comparison DataFrame from the last step of each method's history.
    Avoids re-running evaluation (LL, KL, structural error, etc.); uses the values
    already computed at the final pruning step.
    """
    method_order = method_order or DEFAULT_METHOD_ORDER
    # Column mapping: history key -> comparison table column name
    key_map = {
        "ll_score": "neg_ll",
        "kl_score": "kl_div",
        "structure_score": "structural_error",
        "pred_accuracy": "pred_accuracy",
        "collider_recall": "collider_recall",
        "collider_precision": "collider_precision",
        "global_ace_diff": "global_ace_diff",
        "interventional_kl_mean": "interventional_kl_mean",
    }
    rows = []
    for name in method_order:
        if name not in histories_dict:
            continue
        history = histories_dict[name]
        if not history:
            continue
        last = history[-1]
        r = {"method": name}
        for hkey, ckey in key_map.items():
            if hkey in last:
                val = last[hkey]
                if val is None and ckey in ("pred_accuracy", "collider_recall", "collider_precision", "global_ace_diff", "interventional_kl_mean"):
                    val = np.nan
                r[ckey] = val
            else:
                r[ckey] = np.nan
        rows.append(r)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


DEFAULT_METHOD_ORDER = ("Wavelet", "BIC", "AIC", "BDs", "CSI")


def print_comparison(
    df: pd.DataFrame,
    method_col: str = "method",
    predictive_cols: tuple = ("neg_ll", "kl_div", "pred_accuracy"),
    causal_cols: tuple = ("structural_error", "collider_recall", "collider_precision", "interventional_kl_mean", "global_ace_diff"),
    lower_better: tuple = ("neg_ll", "kl_div", "structural_error", "interventional_kl_mean", "global_ace_diff"),
    round_digits: int = 4,
    mark_best: bool = True,
) -> None:
    """Print comparison table with Predictive / Causal sections and best markers (*)."""
    if df is None or len(df) == 0:
        print("No comparison data.")
        return
    df = df.copy()
    if method_col not in df.columns and len(df.columns) > 0:
        df.insert(0, method_col, df.index.astype(str))
    for c in list(predictive_cols) + list(causal_cols):
        if c in df.columns and np.issubdtype(df[c].dtype, np.floating):
            df[c] = df[c].round(round_digits)
    if mark_best:
        for c in list(predictive_cols) + list(causal_cols):
            if c not in df.columns:
                continue
            valid = pd.to_numeric(df[c], errors="coerce")
            if valid.isna().all():
                continue
            best_val = valid.min() if c in lower_better else valid.max()
            df[c] = df[c].astype(object)
            mask = (pd.to_numeric(df[c], errors="coerce") == best_val) & valid.notna()
            df.loc[mask, c] = df.loc[mask, c].astype(str) + " *"
    print("=" * 80)
    print("PRUNING METHODS COMPARISON (Wavelet, BIC, AIC, BDs, CSI)")
    print("=" * 80)
    pred = [c for c in predictive_cols if c in df.columns]
    if pred:
        print("\n--- Predictive quality ---")
        print(df[[method_col] + pred].to_string(index=False))
    caus = [c for c in causal_cols if c in df.columns]
    if caus:
        print("\n--- Causal quality ---")
        print(df[[method_col] + caus].to_string(index=False))
    print("\n" + "-" * 80)
    if mark_best:
        print("* = best in column")
    print("=" * 80)


def run_and_print_comparison(
    true_model: DiscreteBayesianNetwork = None,
    wavelet_model: DiscreteBayesianNetwork = None,
    BIC_model: DiscreteBayesianNetwork = None,
    AIC_model: DiscreteBayesianNetwork = None,
    BDs_model: DiscreteBayesianNetwork = None,
    csi_model: DiscreteBayesianNetwork = None,
    data: pd.DataFrame = None,
    evaluate_data: pd.DataFrame = None,
    target_var: str = None,
    interventions: list = None,
    n_kl: int = 500,
    n_do: int = 300,
    method_order: tuple = None,
    evaluation_module=None,
    histories: dict = None,
) -> pd.DataFrame:
    """
    Print comparison table. If histories is provided, uses the last step of each
    method's history (no re-evaluation). Otherwise builds models dict from the
    five pruned models and runs run_comparison_table.
    """
    method_order = method_order or DEFAULT_METHOD_ORDER
    if histories is not None:
        df = comparison_table_from_histories(histories, method_order=method_order)
        if df is not None and len(df) > 0:
            print_comparison(df)
        return df
    models = {}
    if wavelet_model is not None:
        models["Wavelet"] = wavelet_model
    if BIC_model is not None:
        models["BIC"] = BIC_model
    if AIC_model is not None:
        models["AIC"] = AIC_model
    if BDs_model is not None:
        models["BDs"] = BDs_model
    if csi_model is not None:
        models["CSI"] = csi_model
    if not models:
        print("No models provided.")
        return pd.DataFrame()
    ordered = [(k, models[k]) for k in method_order if k in models]
    models = dict(ordered)
    df = run_comparison_table(
        true_model=true_model,
        models=models,
        data=data,
        evaluate_data=evaluate_data,
        target_var=target_var,
        interventions=interventions,
        n_kl=n_kl,
        n_do=n_do,
        verbose=True,
        evaluation_module=evaluation_module,
    )
    print_comparison(df)
    return df


# ---------------------------------------------------------------------------
# Plot progress by step for all methods
# ---------------------------------------------------------------------------

# Metrics to plot (keys in history dicts). Order and labels for plots.
PROGRESS_METRICS = (
    ("num_edges", "Number of edges", "lower"),
    ("ll_score", "Log-likelihood (hold-out)", "higher"),
    ("kl_score", "KL divergence (to true)", "lower"),
    ("structure_score", "Structural error (CI disagreement)", "lower"),
    ("pred_accuracy", "Prediction accuracy", "higher"),
    ("global_ace_diff", "Global ACE difference", "lower"),
    ("collider_recall", "Collider recall", "higher"),
    ("collider_precision", "Collider precision", "higher"),
    ("interventional_kl_mean", "Interventional KL (mean)", "lower"),
)


def histories_to_dataframe(histories_dict):
    """
    Convert {method_name: history_list} to a long DataFrame with columns:
    method, step, num_edges, ll_score, kl_score, structure_score, ...
    """
    rows = []
    for method_name, history in histories_dict.items():
        if not history:
            continue
        for rec in history:
            row = {"method": method_name}
            step = rec.get("step")
            if step is None:
                continue
            row["step"] = step
            for key in ("num_edges", "ll_score", "kl_score", "structure_score",
                       "pred_accuracy", "collider_recall", "collider_precision",
                       "global_ace_diff", "interventional_kl_mean", "score"):
                if key in rec:
                    val = rec[key]
                    if isinstance(val, (list, tuple)) and len(val) == 0:
                        continue
                    row[key] = val
            rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def plot_pruning_progress(
    histories_dict,
    metrics=None,
    method_order=None,
    figsize_per_plot=(6, 3.5),
    ncols=2,
    sharex=True,
):
    """
    Plot scoring quality vs pruning step for all methods.

    Parameters
    ----------
    histories_dict : dict
        {"Wavelet": wavelet_history, "BIC": BIC_history, "AIC": AIC_history,
         "BDs": BDs_history, "CSI": csi_history}. Pass only the methods you have.
    metrics : list of str or None
        Keys to plot (e.g. ["ll_score", "kl_score", "structure_score", "num_edges"]).
        If None, uses all keys from PROGRESS_METRICS that appear in the data.
    method_order : tuple or None
        Order of methods in the legend (default: Wavelet, BIC, AIC, BDs, CSI).
    figsize_per_plot : tuple
        (width, height) per subplot.
    ncols : int
        Number of subplot columns.
    sharex : bool
        Share x-axis across subplots.

    Returns
    -------
    fig, axes
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for plot_pruning_progress. Install with: pip install matplotlib")
        return None, None

    df = histories_to_dataframe(histories_dict)
    if df.empty:
        print("No history data to plot.")
        return None, None

    method_order = method_order or DEFAULT_METHOD_ORDER
    methods = [m for m in method_order if m in df["method"].values]
    if not methods:
        methods = list(df["method"].unique())

    metric_info = {m[0]: (m[1], m[2]) for m in PROGRESS_METRICS}
    if metrics is None:
        metrics = [m[0] for m in PROGRESS_METRICS if m[0] in df.columns and m[0] != "score"]
    else:
        metrics = [m for m in metrics if m in df.columns]

    if not metrics:
        print("No plottable metrics found in history. Available:", list(df.columns))
        return None, None

    nplots = len(metrics)
    nrows = (nplots + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_plot[0] * ncols, figsize_per_plot[1] * nrows),
        sharex=sharex,
        squeeze=False,
    )
    axes = axes.flatten()

    colors = plt.cm.tab10(np.linspace(0, 1, max(len(methods), 10)))
    method_color = {m: colors[i % len(colors)] for i, m in enumerate(methods)}

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        title, _ = metric_info.get(metric, (metric.replace("_", " ").title(), ""))
        for method in methods:
            sub = df[df["method"] == method].sort_values("step")
            if sub.empty or metric not in sub.columns:
                continue
            y = pd.to_numeric(sub[metric], errors="coerce")
            if y.isna().all():
                continue
            ax.plot(
                sub["step"],
                y,
                marker="o",
                markersize=4,
                label=method,
                color=method_color.get(method, None),
            )
        ax.set_title(title)
        ax.set_xlabel("Pruning step")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

    for j in range(len(metrics), len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    return fig, axes
