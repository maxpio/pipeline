import os
import sys
from pyscipopt import Model, SCIP_PARAMSETTING

# Allow importing from same package
_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SRC_DIR)
from feature_extractor import get_master_constraints
"""
Constructs and manages a Lagrangian relaxation of a MIP model using PySCIPOpt.
"""
class LagrangianRelaxation:
    """Manages the Lagrangian relaxation of an LP/MIP model."""
    def __init__(self, lp_file_path, dec_file_path, disable_presolve=False, disable_heuristics=False, mip_gap=0.0):
        """Initializes the Lagrangian relaxation of the given model."""
        # Resolve lp path
        if not os.path.isabs(lp_file_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)
            lp_file_path = os.path.join(base_dir, lp_file_path)

        # Initialize SCIP model
        self.model = Model("Lagrangian_Relaxation")
        self.model.hideOutput()
        
        if disable_presolve:
            self.model.setPresolve(SCIP_PARAMSETTING.OFF)
            
        if disable_heuristics:
            self.model.setHeuristics(SCIP_PARAMSETTING.OFF)
            
        if mip_gap > 0.0:
            self.model.setRealParam("limits/gap", mip_gap)

        self.model.readProblem(lp_file_path)
        
        self.orig_sense = self.model.getObjectiveSense()
        
        self.master_conss = get_master_constraints(dec_file_path)
        self.hard_conss = {}
        self.multipliers = {}
        
        # Store original objective
        self.orig_obj = {v.name: v.getObj() for v in self.model.getVars()}
        
        self._extract_and_relax_hard_constraints()

    def _extract_and_relax_hard_constraints(self):
        """Extracts and relaxes master constraints (from .dec file)."""
        conss = self.model.getConss()
        for cons in conss:
            if cons.name in self.master_conss:
                lhs = self.model.getLhs(cons)
                rhs = self.model.getRhs(cons)
                
                coeffs = {}
                try:
                    # Extract coefficients
                    vars_in_cons = self.model.getConsVars(cons)
                    vals_in_cons = self.model.getConsVals(cons)
                    for v, val in zip(vars_in_cons, vals_in_cons):
                        coeffs[v.name] = val
                except AttributeError as e:
                    raise RuntimeError(f"Could not extract coefficients for '{cons.name}' directly. "
                                       f"Ensure your PySCIPOpt build supports getConsVars/getConsVals.") from e
                
                # Store formulation
                self.hard_conss[cons.name] = {
                    "coeffs": coeffs,
                    "lhs": lhs,
                    "rhs": rhs
                }
                
                # Init multiplier
                self.multipliers[cons.name] = 0.0
                
                # Relax bounds
                self.model.chgLhs(cons, -self.model.infinity())
                self.model.chgRhs(cons, self.model.infinity())

    def set_multipliers(self, multipliers_dict):
        """Updates Lagrangian multipliers and objective function."""
        # Reset stage
        if self.model.getStage() != "problem":
            self.model.freeTransform()
            
        for name, val in multipliers_dict.items():
            if name in self.multipliers:
                self.multipliers[name] = val
            else:
                print(f"Warning: Constraint '{name}' not found among hard constraints.")
                
        self._update_objective()
        
    def _update_objective(self):
        """Recalculates and sets the new objective expression based on multipliers."""
        # Copy objective
        new_obj_coeffs = {v_name: obj for v_name, obj in self.orig_obj.items()}
        inf = self.model.infinity()
        
        # Track offset
        lagrangian_offset = 0.0
        
        # Determine sign
        sense_sign = 1 if self.orig_sense == "minimize" else -1

        for c_name, c_data in self.hard_conss.items():
            lam = self.multipliers[c_name]
            if lam == 0.0:
                continue
                
            coeffs = c_data["coeffs"]
            rhs = c_data["rhs"]
            lhs = c_data["lhs"]
            
            # <= constraint
            if rhs < inf:
                lagrangian_offset -= sense_sign * lam * rhs
                for v_name, coef in coeffs.items():
                    new_obj_coeffs[v_name] += sense_sign * lam * coef
            
            # >= constraint
            elif lhs > -inf:
                lagrangian_offset += sense_sign * lam * lhs
                for v_name, coef in coeffs.items():
                    new_obj_coeffs[v_name] -= sense_sign * lam * coef
                    
        # Reconstruct objective
        new_obj_expr = 0.0
        var_dict = {v.name: v for v in self.model.getVars()}
        for v_name, new_coef in new_obj_coeffs.items():
            if new_coef != 0.0 and v_name in var_dict:
                new_obj_expr += new_coef * var_dict[v_name]
                
        # Set objective
        self.model.setObjective(new_obj_expr, sense=self.orig_sense)
        
        # Store offset
        self.lagrangian_offset = lagrangian_offset
        
    def optimize_and_get_violations(self):
        """Optimizes the relaxed model and computes violations of master constraints."""
        self.model.optimize()
        status = self.model.getStatus()
        
        if status not in ("optimal", "gaplimit"):
            print(f"Model did not solve to optimality. Status: {status}")
            return None, None
            
        # Get bound
        bound = self.model.getDualbound() + getattr(self, "lagrangian_offset", 0.0)
        
        # Calculate violations
        var_dict = {v.name: v for v in self.model.getVars()}
        violations = {}
        inf = self.model.infinity()
        
        for c_name, c_data in self.hard_conss.items():
            coeffs = c_data["coeffs"]
            rhs = c_data["rhs"]
            lhs = c_data["lhs"]
            
            # Calculate dot product
            ax_val = sum(coef * self.model.getVal(var_dict[v_name]) 
                         for v_name, coef in coeffs.items() if v_name in var_dict)
            
            violation = 0.0
            
            # <= violation
            if rhs < inf:
                violation += (ax_val - rhs)
                
            # >= violation
            elif lhs > -inf:
                violation += (lhs - ax_val)
                
            violations[c_name] = violation
            
        return bound, violations