"""
generate_test_configs.py
------------------------
Generates experiment YAML files in the tests_dir read from config_data.yaml.

Each generated file only contains parameters that DIFFER from config_test_base.yaml,
but always contains at least one parameter so no file is empty.

Combination dimensions
======================
1. raw  (boolean)
   True  → disable_heuristics/presolving/separating_cliques=True,
            masterpricer_stabilization=False  → append "rw"
   False → those four params flipped              → append "dflt"

2. duals (3 options)
   opt   → dualvalue_type="optimal",          use_custom_duals=True  → append "o"
   pred  → dualvalue_type="predicted",        use_custom_duals=True  → append "p"
   none  → dualvalue_type="random",           use_custom_duals=False → append ""

3. master_smoothing (boolean)
   True  → masterpricer_stabilization=True    → append "msth"
            (only allowed when use_smoothing is False)
   False → masterpricer_stabilization=False   → append ""

4. init_columns (boolean)
   True  → add_round_0_columns=True           → append "intc"
            (only allowed when use_custom_duals is True)
   False → add_round_0_columns=False          → append ""

5. pert_columns (boolean)
   True  → n_perturbation_rounds=20           → append "pertc"
            (only allowed when use_custom_duals is True)
   False → n_perturbation_rounds=0            → append ""

6. dual_smoothing (boolean)
   True  → use_smoothing=True                 → append "dsth"
            (only allowed when use_custom_duals is True)
   False → use_smoothing=False                → append ""

7. subprob_presolve_heur (boolean)
   True  → disable_subprob_presolve_heur=True → append ""
   False → disable_subprob_presolve_heur=False → append "pHpr"

Constraint summary
------------------
- master_smoothing=True  requires use_smoothing=False  (i.e. dual_smoothing=False)
- init_columns=True      requires use_custom_duals=True (i.e. duals != "none")
- pert_columns=True      requires use_custom_duals=True (i.e. duals != "none")
- dual_smoothing=True    requires use_custom_duals=True (i.e. duals != "none")
"""

import itertools
import os
import sys

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)

CONFIG_DATA_PATH = os.path.join(REPO_ROOT, "config", "config_data.yaml")
CONFIG_BASE_PATH = os.path.join(REPO_ROOT, "config", "config_test_base.yaml")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def get_tests_dir() -> str:
    data = load_yaml(CONFIG_DATA_PATH)
    try:
        return data["orchestrator_settings"]["tests_dir"]
    except KeyError as exc:
        raise RuntimeError(
            f"Could not find orchestrator_settings.tests_dir in {CONFIG_DATA_PATH}"
        ) from exc


def flatten_base(base: dict) -> dict:
    """Return a flat {dotted.key: value} dict for the base config."""
    result = {}

    def _recurse(d, prefix):
        for k, v in d.items():
            full = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                _recurse(v, full)
            else:
                result[full] = v

    _recurse(base, "")
    return result


def build_diff(overrides: dict, base_flat: dict) -> dict:
    """
    Given a flat {dotted.key: value} overrides dict, return a nested dict
    containing only entries that differ from the base.  Always returns at
    least one key (the first override) so files are never empty.
    """
    different = {}
    for key, val in overrides.items():
        if base_flat.get(key) != val:
            different[key] = val

    if not different:
        # Fallback: include the very first override to ensure a non-empty file.
        first_key = next(iter(overrides))
        different[first_key] = overrides[first_key]

    # Reconstruct nested dict
    nested: dict = {}
    for dotted, val in different.items():
        parts = dotted.split(".")
        cur = nested
        for part in parts[:-1]:
            cur = cur.setdefault(part, {})
        cur[parts[-1]] = val

    return nested


def dump_yaml(data: dict, path: str) -> None:
    """Write *data* to *path* with clean YAML formatting."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ---------------------------------------------------------------------------
# Dimension definitions
# ---------------------------------------------------------------------------

# Base key prefix for all gcg_settings
GCG = "prediction_parameters.gcg_settings"
PP = "prediction_parameters"

BASE_KEYS = {
    # Top-level prediction params
    f"{PP}.dualvalue_type": "optimal",
    f"{PP}.experiment_json_name": "opt.json",
    # gcg_settings defaults (from config_test_base.yaml)
    f"{GCG}.use_custom_duals": True,
    f"{GCG}.disable_heuristics": True,
    f"{GCG}.disable_presolving": True,
    f"{GCG}.disable_separating_cliques": True,
    f"{GCG}.disable_subprob_presolve_heur": True,
    f"{GCG}.masterpricer_stabilization": False,
    f"{GCG}.use_smoothing": True,
    f"{GCG}.n_perturbation_rounds": 0,
    f"{GCG}.add_round_0_columns": False,
}

# Dim 1: raw
RAW_OPTIONS = [
    (
        True,
        "rw",
        {
            f"{GCG}.disable_heuristics": True,
            f"{GCG}.disable_presolving": True,
            f"{GCG}.disable_separating_cliques": True,
            f"{GCG}.masterpricer_stabilization": False,
        },
    ),
    (
        False,
        "dflt",
        {
            f"{GCG}.disable_heuristics": False,
            f"{GCG}.disable_presolving": False,
            f"{GCG}.disable_separating_cliques": False,
            f"{GCG}.masterpricer_stabilization": True,
        },
    ),
]

# Dim 2: duals
DUALS_OPTIONS = [
    ("opt",  "o", "optimal",             True),
    ("pred", "p",   "predicted",           True),
    ("none", "",    "random",              False),
]

# Dim 3: master smoothing (True only if use_smoothing is False, i.e. dual_smoothing=False)
MASTER_SMOOTH_OPTIONS = [True, False]

# Dim 4: init columns (True only if use_custom_duals=True)
INIT_COL_OPTIONS = [True, False]

# Dim 5: pert columns (True only if use_custom_duals=True)
PERT_COL_OPTIONS = [True, False]

# Dim 6: dual smoothing (True only if use_custom_duals=True)
DUAL_SMOOTH_OPTIONS = [True, False]

# Dim 7: subprob presolve/heur (False → pHpr suffix)
SUBPROB_OPTIONS = [True, False]


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def generate_combinations():
    """Yield (name_suffix, overrides_flat) for every valid combination."""

    for (raw_val, raw_sfx, raw_params), \
        (dual_key, dual_sfx, dual_type, use_custom), \
        master_smooth, init_col, pert_col, dual_smooth, subprob \
        in itertools.product(
            RAW_OPTIONS,
            DUALS_OPTIONS,
            MASTER_SMOOTH_OPTIONS,
            INIT_COL_OPTIONS,
            PERT_COL_OPTIONS,
            DUAL_SMOOTH_OPTIONS,
            SUBPROB_OPTIONS,
        ):

        # ---- Constraint checks ----
        # master_smoothing=True only when dual_smoothing=False (use_smoothing=False)
        if master_smooth and dual_smooth:
            continue

        # init_col, pert_col, dual_smooth only when use_custom_duals=True
        if init_col and not use_custom:
            continue
        if pert_col and not use_custom:
            continue
        if dual_smooth and not use_custom:
            continue

        # "dflt should only be tested without additional duals."
        if (raw_val is False) and use_custom:
            continue

        # "rw with one dual file used should imply that at least one setting is actually active using the duals"
        if (raw_val is True) and use_custom:
            if not (dual_smooth or pert_col or init_col):
                continue

        # If duals is "none" (use_custom=False), only two specific combinations are allowed:
        # 1) "full default with mastersmoothing and pricing prob heuristics activated"
        # 2) "raw with both options deactivated"
        if not use_custom:
            # Combination 1: dflt_msth_pHpr
            is_dflt_msth_phpr = (raw_val is False) and (master_smooth is True) and (subprob is False)
            # Combination 2: rw
            is_rw = (raw_val is True) and (master_smooth is False) and (subprob is True)
            
            if not (is_dflt_msth_phpr or is_rw):
                continue

        # masterpricer_stabilization: dim1 sets it, dim3 can override to True
        # but dim1 "dflt" already sets masterpricer_stabilization=True,
        # so master_smooth=True on top of dflt is redundant — still generate it
        # (unless that conflicts with use_smoothing).
        # Per spec: master_smooth True only if use_smoothing is False (dual_smooth=False)
        # Already handled above.

        # ---- Build name ----
        name_parts = []
        if dual_sfx:
            name_parts.append(dual_sfx)     # "opt", "sub", "rnd", or ""
        if master_smooth:
            name_parts.append("msth")
        if init_col:
            name_parts.append("intc")
        if pert_col:
            name_parts.append("pertc")
        if dual_smooth:
            name_parts.append("dsth")
        if not subprob:                     # disabled → add "pHpr"
            name_parts.append("pHpr")

        # Only append "rw" if there are no other parts.
        if raw_sfx == "rw":
            if not name_parts:
                name_parts.insert(0, raw_sfx)
        elif raw_sfx:
            name_parts.insert(0, raw_sfx)

        name = "_".join(p for p in name_parts if p)

        # ---- Build overrides ----
        overrides: dict = {}

        # Dim 1: raw
        overrides.update(raw_params)

        # Dim 2: duals
        overrides[f"{PP}.dualvalue_type"] = dual_type
        overrides[f"{GCG}.use_custom_duals"] = use_custom

        # Dim 3: master smoothing
        # If True, sets masterpricer_stabilization=True and adds "msth".
        # If False, sets masterpricer_stabilization=False (overriding Dim 1 if it flipped it to True).
        overrides[f"{GCG}.masterpricer_stabilization"] = master_smooth

        # Dim 4: init columns
        overrides[f"{GCG}.add_round_0_columns"] = init_col

        # Dim 5: pert columns
        overrides[f"{GCG}.n_perturbation_rounds"] = 20 if pert_col else 0

        # Dim 6: dual smoothing
        overrides[f"{GCG}.use_smoothing"] = dual_smooth

        # Dim 7: subprob presolve/heur
        overrides[f"{GCG}.disable_subprob_presolve_heur"] = subprob

        # Experiment JSON name derived from the name
        overrides[f"{PP}.experiment_json_name"] = f"{name}.json"

        yield name, overrides


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tests_dir = get_tests_dir()
    base = load_yaml(CONFIG_BASE_PATH)
    base_flat = flatten_base(base)

    print(f"Tests directory : {tests_dir}")
    print(f"Base config     : {CONFIG_BASE_PATH}")
    print()

    generated = 0
    skipped_duplicates = set()

    for name, overrides_flat in generate_combinations():
        if name in skipped_duplicates:
            # Two different constraint paths produced the same name → skip duplicates
            print(f"  [SKIP duplicate] {name}.yaml")
            continue
        skipped_duplicates.add(name)

        diff = build_diff(overrides_flat, base_flat)
        out_path = os.path.join(tests_dir, f"{name}.yaml")

        dump_yaml(diff, out_path)
        generated += 1
        print(f"  [OK] {name}.yaml  ({len(diff.get('prediction_parameters', diff))} top-level keys in diff)")

    print()
    print(f"Generated {generated} experiment config files in: {tests_dir}")


if __name__ == "__main__":
    main()
