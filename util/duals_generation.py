"""
Generates optimal, suboptimal, and random dual values (multipliers) for Lagrangian relaxation of MILP instances.
Reads LP files, relaxes master constraints (from .dec files) and uses a subgradient method to find dual bounds.
"""
import os
import sys
import glob
import json
import random
import concurrent.futures
import yaml
from pyscipopt import Model

# Allow importing from src_ml
_BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_BASE, "..", "src_ml"))
from feature_extractor import get_master_constraints

# Paths and Settings
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
config_full = {}
for conf_file in ["config/config_general.yaml", "config/config_data.yaml"]:
    config_path = os.path.abspath(os.path.join(BASE_DIR, "..", conf_file))
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config_full.update(yaml.safe_load(f))

config = config_full.get("duals_generation_settings", {})

DATA_DIR = config_full.get("general_settings", {}).get("data_dir")
if not DATA_DIR:
    raise ValueError("general_settings -> data_dir not found in config.yaml")

LP_DIR = os.path.join(DATA_DIR, "lpfiles")
DEC_DIR = os.path.join(DATA_DIR, "decfiles")
DUAL_BASE_DIR = os.path.join(DATA_DIR, "dualvalues")

DUAL_VALUES_DIR = os.path.join(DUAL_BASE_DIR, "optimal")
RANDOM_DUAL_VALUES_DIR = os.path.join(DUAL_BASE_DIR, "random")

BOUNDS_DIR = os.path.join(DATA_DIR, "bounds")
RESULTS_JSON_FILE = os.path.join(BOUNDS_DIR, "lagrangian_bounds.json")

# Generation Options
PROCESS_SPLITS = config.get("process_splits", False)
PREFIX_FILTER = config.get("prefix_filter", "gap")
GENERATE_SUBOPT = config.get("generate_subopt", True)
SUBOPT_PERCENT = config.get("subopt_percent", 0.995)

_subopt_str = str(SUBOPT_PERCENT).split('.')[-1] if '.' in str(SUBOPT_PERCENT) else str(SUBOPT_PERCENT)
SUBOPT_DUAL_VALUES_DIR = os.path.join(DUAL_BASE_DIR, f"suboptDualvalues{_subopt_str}")

GENERATE_RANDOM = config.get("generate_random", True)
RANDOM_GAUSS_SIGMA = config.get("random_gauss_sigma", 5.0)

# Execution Options
MULTICORE_SOLVING = config.get("multicore_solving", False)
MAX_WORKERS = config_full.get("general_settings", {}).get("num_cores", 1)
SHORT_LOG_OUTPUT = MULTICORE_SOLVING

# Subgradient Params
MAX_ITERS = config.get("max_iters", 1000)
INITIAL_STEP_SIZE = config.get("initial_step_size", 4.0)
NO_IMPROVE_LIMIT = config.get("no_improve_limit", 10)
STEP_SIZE_SHRINK = config.get("step_size_shrink", 0.5)
TOLERANCE = float(config.get("tolerance", 1e-6))

class LagrangianRelaxation:
    """Manages the Lagrangian relaxation of an LP model."""
    def __init__(self, lp_file_path, dec_file_path):
        """Initializes the model from an LP file and relaxes master constraints from the .dec file."""
        self.model = Model("Lagrangian_Relaxation")
        self.model.hideOutput()
        self.model.readProblem(lp_file_path)
        self.orig_sense = self.model.getObjectiveSense()
        
        self.master_conss = get_master_constraints(dec_file_path)
        self.hard_conss = {}
        self.multipliers = {}
        self.orig_obj = {v.name: v.getObj() for v in self.model.getVars()}
        self._extract_and_relax_hard_constraints()

    def _extract_and_relax_hard_constraints(self):
        """Identifies master constraints (from .dec file), extracts their data, and relaxes their bounds in the model."""
        for cons in self.model.getConss():
            if cons.name in self.master_conss:
                lhs, rhs = self.model.getLhs(cons), self.model.getRhs(cons)
                coeffs = {}
                try:
                    for v, val in zip(self.model.getConsVars(cons), self.model.getConsVals(cons)):
                        coeffs[v.name] = val
                except AttributeError:
                    print(f"Warning: Could not extract coefficients for '{cons.name}'.")

                self.hard_conss[cons.name] = {"coeffs": coeffs, "lhs": lhs, "rhs": rhs}
                self.multipliers[cons.name] = 0.0
                
                self.model.chgLhs(cons, -self.model.infinity())
                self.model.chgRhs(cons, self.model.infinity())

    def set_multipliers(self, multipliers_dict):
        """Updates the Lagrangian multipliers and recalculates the objective function."""
        if self.model.getStage() != "problem":
            self.model.freeTransform()
        
        self.multipliers.update(multipliers_dict)
        self._update_objective()

    def _update_objective(self):
        """Updates the model's objective function based on the current multipliers."""
        new_obj_coeffs = self.orig_obj.copy()
        inf = self.model.infinity()
        self.lagrangian_offset = 0.0
        sense_sign = 1 if self.orig_sense == "minimize" else -1

        for c_name, c_data in self.hard_conss.items():
            lam = self.multipliers[c_name]
            if lam == 0.0: continue
            
            coeffs, rhs, lhs = c_data["coeffs"], c_data["rhs"], c_data["lhs"]
            
            if rhs < inf:
                self.lagrangian_offset -= sense_sign * lam * rhs
                for v_name, coef in coeffs.items():
                    new_obj_coeffs[v_name] += sense_sign * lam * coef
            elif lhs > -inf:
                self.lagrangian_offset += sense_sign * lam * lhs
                for v_name, coef in coeffs.items():
                    new_obj_coeffs[v_name] -= sense_sign * lam * coef

        new_obj_expr = 0.0
        var_dict = {v.name: v for v in self.model.getVars()}
        for v_name, new_coef in new_obj_coeffs.items():
            if new_coef != 0.0 and v_name in var_dict:
                new_obj_expr += new_coef * var_dict[v_name]
                
        self.model.setObjective(new_obj_expr, sense=self.orig_sense)

    def optimize_and_get_violations(self):
        """Optimizes the relaxed model and returns the dual bound and constraint violations."""
        self.model.optimize()
        if self.model.getStatus() not in ("optimal", "gaplimit"):
            return None, None
            
        bound = self.model.getDualbound() + getattr(self, "lagrangian_offset", 0.0)
        var_dict = {v.name: v for v in self.model.getVars()}
        violations, inf = {}, self.model.infinity()
        
        for c_name, c_data in self.hard_conss.items():
            ax_val = sum(coef * self.model.getVal(var_dict[v_name]) 
                         for v_name, coef in c_data["coeffs"].items() if v_name in var_dict)
            violation = 0.0
            if c_data["rhs"] < inf:
                violation += (ax_val - c_data["rhs"])
            elif c_data["lhs"] > -inf:
                violation += (c_data["lhs"] - ax_val)
                
            violations[c_name] = violation
            
        return bound, violations

def solve_with_subgradient(lr_model, target_opt=None, subopt_percent=None):
    """Executes the subgradient method to optimize the Lagrangian multipliers."""
    multipliers = {cons_name: 0.0 for cons_name in lr_model.hard_conss}
    best_dual_bound = -float('inf') if lr_model.orig_sense == "minimize" else float('inf')
    best_multipliers = multipliers.copy()
    step_size = INITIAL_STEP_SIZE
    no_improve_iters = 0
    history = []

    if not SHORT_LOG_OUTPUT:
        print(f"{'Iter':>4} | {'Obj Value':>15} | {'Step Size':>10} | {'Sum Sq. Grad':>15}")
    for k in range(1, MAX_ITERS + 1):
        lr_model.set_multipliers(multipliers)
        obj_val, subgradients = lr_model.optimize_and_get_violations()
        
        if obj_val is None: break

        improved = (obj_val > best_dual_bound) if lr_model.orig_sense == "minimize" else (obj_val < best_dual_bound)
        if improved:
            best_dual_bound, best_multipliers = obj_val, multipliers.copy()
            no_improve_iters = 0
            history.append((best_dual_bound, best_multipliers.copy()))
            
            if target_opt is not None and subopt_percent is not None:
                if abs(best_dual_bound - target_opt) <= abs(target_opt) * (1.0 - subopt_percent) or \
                   (lr_model.orig_sense == "minimize" and best_dual_bound >= target_opt) or \
                   (lr_model.orig_sense == "maximize" and best_dual_bound <= target_opt):
                    if not SHORT_LOG_OUTPUT:
                        print(f"Target dual bound proximity reached! Stopping early.")
                    break
        else:
            no_improve_iters += 1

        if no_improve_iters >= NO_IMPROVE_LIMIT:
            step_size *= STEP_SIZE_SHRINK
            no_improve_iters = 0

        sum_sq_grad = sum(g**2 for g in subgradients.values())
        if not SHORT_LOG_OUTPUT:
            print(f"{k:4d} | {obj_val:15.4f} | {step_size:10.4f} | {sum_sq_grad:15.4f}")
        
        if sum_sq_grad < TOLERANCE: break

        for c_name, c_data in lr_model.hard_conss.items():
            new_lam = multipliers[c_name] + step_size * subgradients[c_name]
            multipliers[c_name] = new_lam if c_data["lhs"] == c_data["rhs"] else max(0.0, new_lam)

    return best_dual_bound, best_multipliers, history

def process_instance(lp_file):
    """Processes a single LP instance: finds optimal, suboptimal, and random dual values."""
    filename = os.path.basename(lp_file)
    base_name = os.path.splitext(filename)[0]
    dec_file = os.path.join(DEC_DIR, f"{base_name}.dec")
    
    if not SHORT_LOG_OUTPUT:
        print(f"\n{'='*40}")
        print(f"Processing {filename}...")
        print(f"{'='*40}")
    
    if not os.path.exists(dec_file):
        if not SHORT_LOG_OUTPUT:
            print(f"No .dec file found for {filename} at {dec_file}. Skipping...")
        return filename, None
        
    lr_model = LagrangianRelaxation(lp_file, dec_file)
    
    if not lr_model.hard_conss:
        if not SHORT_LOG_OUTPUT:
            print(f"No master constraints found in {filename}. Skipping...")
        return filename, None
        
    best_obj, optimal_multipliers, _ = solve_with_subgradient(lr_model)
    
    if not SHORT_LOG_OUTPUT:
        print(f"Optimal Dual Objective: {best_obj:.4f}")
        
    out_file_opt = os.path.join(DUAL_VALUES_DIR, f"{base_name}.txt")
    with open(out_file_opt, "w") as f:
        for cons_name, dual_val in optimal_multipliers.items():
            f.write(f'"{cons_name}": {dual_val},\n')
    if not SHORT_LOG_OUTPUT:
        print(f"Saved optimal multipliers to: {out_file_opt}")

    if GENERATE_SUBOPT:
        if not SHORT_LOG_OUTPUT:
            print(f"--- Running to find Suboptimal Multipliers (Target: within {SUBOPT_PERCENT*100}% of {best_obj:.4f}) ---")
        _, subopt_multipliers, _ = solve_with_subgradient(lr_model, target_opt=best_obj, subopt_percent=SUBOPT_PERCENT)
        
        out_file_subopt = os.path.join(SUBOPT_DUAL_VALUES_DIR, f"{base_name}.txt")
        with open(out_file_subopt, "w") as f:
            for cons_name, dual_val in subopt_multipliers.items():
                f.write(f'"{cons_name}": {dual_val},\n')
        if not SHORT_LOG_OUTPUT:
            print(f"Saved suboptimal multipliers to: {out_file_subopt}")

    if GENERATE_RANDOM:
        is_duals_positive = False
        for val in optimal_multipliers.values():
            if val > 1e-6:
                is_duals_positive = True
                break
            elif val < -1e-6:
                is_duals_positive = False
                break

        out_file = os.path.join(RANDOM_DUAL_VALUES_DIR, f"{base_name}.txt")
        with open(out_file, "w") as f:
            for cons_name, dual_val in optimal_multipliers.items():
                perturbed = random.gauss(dual_val, RANDOM_GAUSS_SIGMA)
                
                if is_duals_positive:
                    while perturbed < 0:
                        perturbed = random.gauss(dual_val, RANDOM_GAUSS_SIGMA)
                else:
                    while perturbed > 0:
                        perturbed = random.gauss(dual_val, RANDOM_GAUSS_SIGMA)
                    is_eq = lr_model.hard_conss[cons_name]["lhs"] == lr_model.hard_conss[cons_name]["rhs"]
                    if not is_eq:
                        perturbed = max(0.0, perturbed)
                
                f.write(f'"{cons_name}": {perturbed},\n')
        if not SHORT_LOG_OUTPUT:
            print(f"Saved random perturbed multipliers to: {out_file}")
            
    if not SHORT_LOG_OUTPUT:
        print(f"\n--- Finished {filename} ---")
        
    return base_name, best_obj

if __name__ == "__main__":
    os.makedirs(DUAL_VALUES_DIR, exist_ok=True)
    if GENERATE_SUBOPT:
        os.makedirs(SUBOPT_DUAL_VALUES_DIR, exist_ok=True)
    if GENERATE_RANDOM:
        os.makedirs(RANDOM_DUAL_VALUES_DIR, exist_ok=True)
        
    if PROCESS_SPLITS:
        lp_files = []
        for split in ['training', 'val', 'test']:
            lp_files.extend(glob.glob(os.path.join(LP_DIR, split, "*.lp")))
        if not lp_files:
            print(f"No .lp files found in subdirectories (training, val, test) of {LP_DIR}")
    else:
        lp_files = glob.glob(os.path.join(LP_DIR, "*.lp"))
        if not lp_files:
            print(f"No .lp files found in {LP_DIR}")
        
    if PREFIX_FILTER:
        lp_files = [f for f in lp_files if os.path.basename(f).startswith(PREFIX_FILTER)]
        
    results = {}
    
    if MULTICORE_SOLVING:
        print(f"Starting multicore solving with {MAX_WORKERS} workers...")
        with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = {executor.submit(process_instance, lp_file): os.path.basename(lp_file) for lp_file in lp_files}
                
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                filename = futures[future]
                completed += 1
                try:
                    fn, best_obj = future.result()
                    if best_obj is not None:
                        results[fn] = best_obj
                        if SHORT_LOG_OUTPUT:
                            print(f"[{completed}/{len(lp_files)}] Finished {fn} (Best Dual Objective: {best_obj:.4f})")
                    else:
                        if SHORT_LOG_OUTPUT:
                            print(f"[{completed}/{len(lp_files)}] Skipped {fn} (No master constraints)")
                except Exception as exc:
                    print(f"[{completed}/{len(lp_files)}] Exception in {filename}: {exc}")
    else:
        for i, lp_file in enumerate(lp_files):
            filename = os.path.basename(lp_file)
            
            fn, best_obj = process_instance(lp_file)
            if best_obj is not None:
                results[fn] = best_obj
                if SHORT_LOG_OUTPUT:
                    print(f"[{i+1}/{len(lp_files)}] Finished {fn} (Best Dual Objective: {best_obj:.4f})")
            else:
                if SHORT_LOG_OUTPUT:
                    print(f"[{i+1}/{len(lp_files)}] Skipped {fn} (No master constraints)")
            
    os.makedirs(os.path.dirname(RESULTS_JSON_FILE), exist_ok=True)
    with open(RESULTS_JSON_FILE, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nSaved optimal dual values summary to {RESULTS_JSON_FILE}")