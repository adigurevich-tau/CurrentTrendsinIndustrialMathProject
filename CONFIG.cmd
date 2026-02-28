# Run configuration

All tunable run parameters are in **`run_config.json`**. Edit that file to change behavior; no code changes needed.

## File to edit / upload

| File | Purpose |
|------|--------|
| **`run_config.json`** | All run configuration (pruning steps, data sizes, evaluation samples). |

If you run from Colab or another environment, upload or create **`run_config.json`** in the project root (same folder as `main.py`). If the file is missing, defaults in `run_config.py` are used.

---

## What you can change (in `run_config.json`)

### Pruning

| Key | Default | Description |
|-----|--------|-------------|
| `pruning.max_steps` | 10 | Maximum number of edges to remove per method (Wavelet, BIC, AIC, BDs, CSI). |
| `pruning.min_steps` | 5 | Minimum steps before early stop for score-based methods (BIC, AIC, BDs). |

### Evaluation (when comparison table is computed by re-evaluation)

| Key | Default | Description |
|-----|--------|-------------|
| `evaluation.n_kl` | 500 | Number of samples for KL divergence estimate (true vs learned). |
| `evaluation.n_do` | 300 | Number of samples per intervention for interventional KL (when interventions are used). |
| `evaluation.target_var` | `null` | Target variable name for prediction accuracy (e.g. `"CATECHOL"` for ALARM). Set to `null` or omit to disable. Must be a node in the BN. |
| `evaluation.interventions` | `[]` | List of intervention dicts for interventional KL, e.g. `[{"HR": 0}, {"CO": 1}]`. Use integer state indices. Empty list `[]` disables. |

### ALARM experiment

| Key | Default | Description |
|-----|--------|-------------|
| `alarm.n_data` | 4200 | Total samples generated from ALARM for fitting. |
| `alarm.n_train` | 900 | Training set size for pruning. |
| `alarm.n_eval` | 900 | Hold-out set size for log-likelihood and metrics. |

### Synthetic experiment

| Key | Default | Description |
|-----|--------|-------------|
| `synthetic.sample_size` | 4000 | Total samples generated from the synthetic BN. |
| `synthetic.train_ratio` | 0.7 | Fraction of samples used for training (rest for evaluation). |
| `synthetic.random_state` | 42 | Random seed for train/eval split (reproducibility). |

---

## Example: fewer steps, smaller data

```json
{
  "pruning": {
    "max_steps": 5,
    "min_steps": 2
  },
  "evaluation": {
    "target_var": "CATECHOL",
    "interventions": [{"HR": 0}, {"CO": 1}]
  },
  "alarm": {
    "n_data": 2000,
    "n_train": 500,
    "n_eval": 500
  }
}
```

To disable target prediction and interventional KL, set `"target_var": null` and `"interventions": []`.

Only include keys you want to override; the rest fall back to defaults.

---

## Other config files

| File | Purpose |
|------|--------|
| **`bayesian_config.json`** | Synthetic BN definition (structure + CPDs). Required only for `python main.py synthetic` or `python main.py all`. |
| **`run_config.json`** | Run tuning (steps, sizes, seeds). Optional; defaults used if missing. |
