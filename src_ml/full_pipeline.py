"""
Full end-to-end pipeline for a single instance:
  1. Extract LP relaxation features from a .lp file (using PySCIPOpt)
  2. Build the bipartite graph and predict dual values with the trained GNN model
  3. Save the predicted duals to a file
  4. Run GCG with the custom pricing plugin that uses the predicted duals

Usage:
    python -m src_ml.full_pipeline <lp_file> <dec_file> [--weights <path>] [--gcg <path>] [--timeout <seconds>]
"""

import os
import sys
import json
import time
import yaml
import argparse
import torch
from pathlib import Path
from pyscipopt import Model as SCIPModel
from torch_geometric.data import Batch

# ---------------------------------------------------------------------------
# Resolve project paths so imports work regardless of cwd
# ---------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(BASE_DIR)

from src_ml.input_graph import create_bipartite_graph
from src_ml.model import LagrangianMultiplierModel
from src_ml.single_runner import run_single_instance

# ---------------------------------------------------------------------------
# Load config.yaml (model architecture, defaults, etc.)
# ---------------------------------------------------------------------------
CONFIG_PATH = os.path.join(BASE_DIR, "config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Model Architecture from config
FEATURE_EMBEDDING_SIZE = config['model_architecture']['feature_embedding_size']
ENCODER_MLP_DIMS = config['model_architecture']['encoder_mlp_dims']
GNN_MLP_DIMS = config['model_architecture']['gnn_mlp_dims']
DECODER_MLP_DIMS = config['model_architecture']['decoder_mlp_dims']
NUM_LAYERS = config['model_architecture']['num_layers']
DROPOUT = config['model_architecture'].get('dropout', 0.0)

DEFAULT_WEIGHTS_PATH = os.path.join(BASE_DIR, config['paths']['weights_path'])
DEFAULT_GCG_EXECUTABLE = config['paths'].get('gcg_executable', "/home/max/gcg/build/gcg-linux-release/bin/gcg")


# ========================== STEP 1: Feature Extraction ==========================

def extract_features_single(lp_file: str) -> dict:
    """
    Extracts LP relaxation features from a single .lp file using PySCIPOpt.
    Returns the feature dictionary (same schema as lp_to_json_general.py output).
    
    The dictionary contains:
      - "variables": { var_name: {w, lpr_val, rc_val, is_int}, ... }
      - "constraints": { cons_name: {rhs, eq, dualized, pi}, ... }
      - "edges": [ {c, v, coeff}, ... ]
    """
    print(f"[Step 1] Extracting LP relaxation features from: {lp_file}")
    start = time.time()

    model = SCIPModel("FeatureExtractor")
    model.hideOutput()
    model.readProblem(str(lp_file))
    model.setIntParam("presolving/maxrounds", config['scip_settings'].get('feature_extraction_presolving_maxrounds', 0))

    variables_dict = {}
    orig_is_int = {}

    for var in model.getVars():
        v_name = var.name
        orig_is_int[v_name] = 1 if (var.isBinary() or var.isIntegral()) else 0
        variables_dict[v_name] = {"w": var.getObj()}
        model.chgVarType(var, "C")

    constraints_dict = {}
    edges_list = []
    inf = model.infinity()

    for cons in model.getConss():
        c_name = cons.name
        lhs = model.getLhs(cons)
        rhs = model.getRhs(cons)

        if lhs == rhs:
            is_eq, c_rhs = 1, rhs
        elif lhs <= -inf and rhs < inf:
            is_eq, c_rhs = 0, rhs
        else:
            raise Exception(f"Illegal constraint type: {c_name}. Only == and <= allowed.")

        constraints_dict[c_name] = {
            "rhs": c_rhs,
            "eq": is_eq,
            "dualized": 1 if "hard" in c_name.lower() else 0
        }

        # Matrix coefficient extraction
        linear_coefs = model.getValsLinear(cons)
        for var_key, coef in linear_coefs.items():
            if coef != 0.0:
                v_name = var_key if isinstance(var_key, str) else var_key.name
                edges_list.append({"c": c_name, "v": v_name, "coeff": coef})

    model.optimize()

    if model.getStatus() == "optimal":
        for var in model.getVars():
            v_name = var.name
            v_data = variables_dict[v_name]
            v_data["lpr_val"] = model.getVal(var)
            try:
                v_data["rc_val"] = model.getVarRedcost(var)
            except:
                v_data["rc_val"] = 0.0
            v_data["is_int"] = orig_is_int[v_name]

        for cons in model.getConss():
            try:
                constraints_dict[cons.name]["pi"] = model.getDualsolLinear(cons)
            except:
                constraints_dict[cons.name]["pi"] = 0.0
    else:
        print(f"  Warning: LP relaxation status is '{model.getStatus()}', features may be incomplete.")
        # Fill in defaults so the graph can still be built
        for v_name in variables_dict:
            variables_dict[v_name].setdefault("lpr_val", 0.0)
            variables_dict[v_name].setdefault("rc_val", 0.0)
            variables_dict[v_name]["is_int"] = orig_is_int[v_name]
        for c_name in constraints_dict:
            constraints_dict[c_name].setdefault("pi", 0.0)

    model.freeProb()
    elapsed = time.time() - start
    print(f"  Feature extraction done in {elapsed:.2f}s  "
          f"({len(variables_dict)} vars, {len(constraints_dict)} cons, {len(edges_list)} edges)")

    return {
        "variables": variables_dict,
        "constraints": constraints_dict,
        "edges": edges_list
    }


# ========================== STEP 2: ML Prediction ==========================

def predict_duals(feature_dict: dict, weights_path: str) -> tuple[dict, str]:
    """
    Builds the bipartite graph from the feature dictionary, loads the trained
    GNN model, and predicts Lagrangian dual multipliers.

    Returns:
        (mult_dict, dual_txt)  — a dict {constraint_name: value} and a
        pre-formatted string ready to be written to a .txt file.
    """
    print(f"[Step 2] Predicting dual values with ML model ({weights_path})")
    start = time.time()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # --- Write feature dict to a temp JSON so create_bipartite_graph can read it ---
    import tempfile
    fd, tmp_json = tempfile.mkstemp(suffix=".json", text=True)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(feature_dict, f)

        # Build graph
        graph_data = create_bipartite_graph(tmp_json).to(device)
    finally:
        if os.path.exists(tmp_json):
            os.remove(tmp_json)

    # --- Load model ---
    model = LagrangianMultiplierModel(
        feature_embedding_size=FEATURE_EMBEDDING_SIZE,
        encoder_mlp_dims=ENCODER_MLP_DIMS,
        gnn_mlp_dims=GNN_MLP_DIMS,
        num_layers=NUM_LAYERS,
        decoder_mlp_dims=DECODER_MLP_DIMS,
        dropout=DROPOUT
    ).to(device)

    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # --- Predict ---
    all_cons_names = list(feature_dict.get("constraints", {}).keys())

    with torch.no_grad():
        batch_data = Batch.from_data_list([graph_data]).to(device)
        lambda_raw, dualized_mask, eq_flags = model(batch_data)

        # Enforce non-negativity for inequality constraints
        actual_multipliers = torch.where(eq_flags == 1.0, lambda_raw, torch.relu(lambda_raw))

    actual_multipliers = actual_multipliers.cpu().tolist()
    mask_list = dualized_mask.cpu().tolist()

    dualized_cons_names = [name for j, name in enumerate(all_cons_names) if mask_list[j]]
    mult_dict = dict(zip(dualized_cons_names, actual_multipliers))

    # Build the text representation matching the format GCG's plugin expects
    lines = []
    for k, v in mult_dict.items():
        lines.append(f'"{k}": {v},')
    dual_txt = "\n".join(lines) + "\n"

    elapsed = time.time() - start
    print(f"  Predicted {len(mult_dict)} dual multipliers in {elapsed:.2f}s")

    return mult_dict, dual_txt


# ========================== STEP 3: Run GCG ==========================

def run_gcg_with_duals(
    lp_file: Path,
    dec_file: Path,
    dual_file: Path,
    gcg_executable: Path,
    experiment_params: dict,
    timeout: int = 3600,
    log_file: Path | None = None,
    save_logs: bool = True,
    print_solver_output: bool = True
) -> tuple[str, float, float, int]:
    """
    Constructs the GCG command string with the custom pricing plugin settings
    and runs the solver.  Returns (status, exec_time, obj_val, mlp_iters).
    """
    print(f"[Step 3] Running GCG with predicted duals")
    print(f"  LP:   {lp_file}")
    print(f"  DEC:  {dec_file}")
    print(f"  DUAL: {dual_file}")

    cmd_string = (
        ("set heuristics emphasis off\n" if experiment_params.get('disable_heuristics', True) else "") +
        ("set presolving maxrounds 0\n" if experiment_params.get('disable_presolving', True) else "") +
        ("set separating clique freq -1\n" if experiment_params.get('disable_separating_cliques', True) else "") +
        ("set limits nodes 1\n" if experiment_params.get('solve_root_only', False) else "") +
        f"set pricingcb initduals duals_file {dual_file}\n"
        f"set pricingcb initduals use_custom_duals {experiment_params.get('use_custom_duals', 'TRUE')}\n"
        f"set pricingcb initduals use_bigm_artificials {experiment_params.get('use_bigm_artificials', 'FALSE')}\n"
        f"set pricingcb initduals n_perturbation_rounds {experiment_params.get('n_perturbation_rounds', 0)}\n"
        f"set pricingcb initduals perturbation_percent {experiment_params.get('perturbation_percent', 0.05)}\n"
        f"set pricingcb initduals add_round_0_columns {experiment_params.get('add_round_0_columns', 'TRUE')}\n"
        f"set pricingcb initduals bigm_value {experiment_params.get('bigm_value', 1000000.0)}\n"
        f"set pricingcb initduals large_log {experiment_params.get('large_log', 'FALSE')}\n"
        f"set pricingcb initduals use_smoothing {experiment_params.get('use_smoothing', 'FALSE')}\n"
        f"set pricingcb initduals smoothing_weight_start {experiment_params.get('smoothing_weight_start', 0.99)}\n"
        f"set pricingcb initduals smoothing_weight_factor {experiment_params.get('smoothing_weight_factor', 0.4)}\n"
        f"set pricingcb initduals smoothing_weight_min {experiment_params.get('smoothing_weight_min', 0.01)}\n"
        f"set pricingcb initduals smoothing_improvement_threshold {experiment_params.get('smoothing_improvement_threshold', 200.0)}\n"
        f"set pricingcb initduals disable_subprob_presolve_heur {experiment_params.get('disable_subprob_presolve_heur', 'FALSE')}\n"
        f"set pricing masterpricer stabilization {experiment_params.get('masterpricer_stabilization', 'FALSE')}\n"
        f"read {lp_file}\n"
        f"read {dec_file}\n"
        "optimize\nquit\n"
    )

    return run_single_instance(
        lp_file=lp_file,
        dec_file=dec_file,
        dual_file=dual_file,
        gcg_executable=gcg_executable,
        cmd_string=cmd_string,
        timeout=timeout,
        log_file=log_file,
        save_logs=save_logs,
        print_solver_output=print_solver_output
    )


# ========================== Main ==========================

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end pipeline: extract features → predict duals → run GCG"
    )
    parser.add_argument("lp_file", type=Path, help="Path to the .lp file")
    parser.add_argument("dec_file", type=Path, help="Path to the .dec file")
    parser.add_argument("--weights", type=Path, default=Path(DEFAULT_WEIGHTS_PATH),
                        help="Path to trained model weights (.pth)")
    parser.add_argument("--gcg", type=Path, default=Path(DEFAULT_GCG_EXECUTABLE),
                        help="Path to the GCG executable")
    parser.add_argument("--timeout", type=int, default=config['pipeline_settings'].get('timeout', 3600),
                        help=f"Max solving time in seconds (default: {config['pipeline_settings'].get('timeout', 3600)})")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Directory for predicted duals and logs (default: next to .lp file)")
    parser.add_argument("--no-logs", action="store_true",
                        help="Do not save GCG log output")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print GCG solver output to console")
    args = parser.parse_args()

    lp_file = args.lp_file.resolve()
    dec_file = args.dec_file.resolve()

    if not lp_file.exists():
        sys.exit(f"Error: LP file not found: {lp_file}")
    if not dec_file.exists():
        sys.exit(f"Error: DEC file not found: {dec_file}")
    if not args.weights.exists():
        sys.exit(f"Error: Model weights not found: {args.weights}")
    if not args.gcg.exists():
        sys.exit(f"Error: GCG executable not found: {args.gcg}")

    instance_name = lp_file.stem
    output_dir = args.output_dir or lp_file.parent
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  Full Pipeline — {instance_name}")
    print("=" * 70)

    # ---- Step 1: Feature extraction ----
    feature_dict = extract_features_single(str(lp_file))

    # ---- Step 2: ML prediction ----
    mult_dict, dual_txt = predict_duals(feature_dict, str(args.weights))

    dual_file = output_dir / f"{instance_name}_predicted_duals.txt"
    with open(dual_file, 'w') as f:
        f.write(dual_txt)
    print(f"  Predicted duals saved to: {dual_file}")

    # ---- Step 3: Run GCG with predicted duals ----
    log_file = output_dir / f"{instance_name}_gcg.log"

    experiment_params = config.get('gcg_settings', {})

    status, exec_time, obj_val, mlp_iters = run_gcg_with_duals(
        lp_file=lp_file,
        dec_file=dec_file,
        dual_file=dual_file,
        gcg_executable=args.gcg,
        experiment_params=experiment_params,
        timeout=args.timeout,
        log_file=log_file,
        save_logs=not args.no_logs,
        print_solver_output=not args.quiet
    )

    # ---- Summary ----
    print("\n" + "=" * 70)
    print("  Pipeline Results")
    print("=" * 70)
    print(f"  Instance:        {instance_name}")
    print(f"  Status:          {status}")
    print(f"  Execution Time:  {exec_time:.2f}s")
    print(f"  Objective Value: {obj_val}")
    print(f"  MLP Iterations:  {mlp_iters}")
    if not args.no_logs:
        print(f"  GCG Log:         {log_file}")
    print(f"  Predicted Duals: {dual_file}")
    print("=" * 70)


if __name__ == '__main__':
    main()
