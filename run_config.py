"""
Load run configuration from run_config.json.
All tunable run parameters (pruning steps, data sizes, evaluation samples) live here.
If run_config.json is missing or a key is absent, defaults below are used.
"""

from __future__ import print_function

import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, "run_config.json")

_DEFAULTS = {
    "pruning": {"max_steps": 10, "min_steps": 5},
    "evaluation": {
        "n_kl": 500,
        "n_do": 300,
        "target_var": None,
        "interventions": [],
    },
    "alarm": {"n_data": 4200, "n_train": 900, "n_eval": 900},
    "synthetic": {"sample_size": 4000, "train_ratio": 0.7, "random_state": 42},
}


def _deep_merge(defaults, loaded):
    """Merge loaded dict into defaults; loaded values override."""
    out = dict(defaults)
    if not isinstance(loaded, dict):
        return out
    for k, v in loaded.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_run_config(path=None):
    """Load run_config.json and return a merged config dict. Use path or CONFIG_PATH."""
    path = path or CONFIG_PATH
    loaded = {}
    if os.path.isfile(path):
        try:
            with open(path, "r") as f:
                loaded = json.load(f)
            # Remove comment key if present (not part of config)
            loaded.pop("comment", None)
        except Exception as e:
            print("[run_config] Could not load {}: {}".format(path, e))
    return _deep_merge(_DEFAULTS, loaded)


# Single global config instance (load once)
run_config = load_run_config()


def get(key, subkey=None, default=None):
    """Get a config value. e.g. get('pruning','max_steps') -> 10."""
    d = run_config.get(key, {})
    if subkey is None:
        return d if d else default
    return d.get(subkey, default)
