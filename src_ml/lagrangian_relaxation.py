import os
from pyscipopt import Model, SCIP_PARAMSETTING
import sys
"""
This module provides a class for constructing and managing a Lagrangian relaxation
of a Mixed Integer Programming (MIP) model using PySCIPOpt. It allows for the
relaxation of "hard" constraints, setting Lagrangian multipliers, updating the
objective function accordingly, and computing the violations (subgradients) of
the relaxed constraints.
"""
class LagrangianRelaxation:
    """
    A class to represent and manage the Lagrangian relaxation of a given LP/MIP model.

    This class reads a model from an LP file, identifies and relaxes constraints
    that are designated as "hard" (by containing "hard" in their names), and provides
    methods to dynamically update the Lagrangian multipliers and optimize the relaxed
    problem to find bounds and subgradients.
    """
    def __init__(self, lp_file_path, disable_presolve=False, disable_heuristics=False, mip_gap=0.0):
        """
        Constructs the Lagrangian relaxation of the model provided in the .lp file.
        Accepts an absolute path or a path relative to the project base directory.
        
        Args:
            lp_file_path (str): Absolute (or relative-to-base) path to the .lp file.
            disable_presolve (bool): Whether to disable SCIP presolving.
            disable_heuristics (bool): Whether to disable SCIP heuristics.
            mip_gap (float): The target relative MIP gap.
        """
        # If the path is not absolute, resolve it relative to the project base dir
        if not os.path.isabs(lp_file_path):
            current_dir = os.path.dirname(os.path.abspath(__file__))
            base_dir = os.path.dirname(current_dir)
            lp_file_path = os.path.join(base_dir, lp_file_path)

        # Initialize and read the SCIP model
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
        

        self.hard_conss = {}
        self.multipliers = {}
        
        # Store original objective coefficients for all variables
        self.orig_obj = {v.name: v.getObj() for v in self.model.getVars()}
        
        self._extract_and_relax_hard_constraints()

    def _extract_and_relax_hard_constraints(self):
        """
        Finds constraints with 'hard' in the name, extracts them, and relaxes their bounds.
        
        This method iterates through all constraints in the model, checks if "hard" is
        in the constraint's name, and if so, stores its coefficients and right/left-hand sides.
        It then initializes the multiplier for the constraint to 0.0 and relaxes the constraint
        by setting its left and right-hand sides to negative and positive infinity, respectively.
        """
        conss = self.model.getConss()
        for cons in conss:
            if "hard" in cons.name:
                lhs = self.model.getLhs(cons)
                rhs = self.model.getRhs(cons)
                
                coeffs = {}
                try:
                    # Attempt to extract variables and coefficients using standard API mapping
                    vars_in_cons = self.model.getConsVars(cons)
                    vals_in_cons = self.model.getConsVals(cons)
                    for v, val in zip(vars_in_cons, vals_in_cons):
                        coeffs[v.name] = val
                except AttributeError as e:
                    raise RuntimeError(f"Could not extract coefficients for '{cons.name}' directly. "
                                       f"Ensure your PySCIPOpt build supports getConsVars/getConsVals.") from e
                
                # Store constraint formulation
                self.hard_conss[cons.name] = {
                    "coeffs": coeffs,
                    "lhs": lhs,
                    "rhs": rhs
                }
                
                # Initialize multiplier to 0
                self.multipliers[cons.name] = 0.0
                
                # Relax the constraint by removing its bounds
                self.model.chgLhs(cons, -self.model.infinity())
                self.model.chgRhs(cons, self.model.infinity())

    def set_multipliers(self, multipliers_dict):
        """
        Updates the Lagrangian multipliers and modifies the objective function.
        
        Args:
            multipliers_dict (dict): A dictionary mapping constraint names to their new
                                     Lagrangian multiplier values (float).
        """
        # SCIP must be in the 'problem' stage to modify the objective. 
        # If it was already optimized, free the transformation.
        if self.model.getStage() != "problem":
            self.model.freeTransform()
            
        for name, val in multipliers_dict.items():
            if name in self.multipliers:
                self.multipliers[name] = val
            else:
                print(f"Warning: Constraint '{name}' not found among hard constraints.")
                
        self._update_objective()
        
    def _update_objective(self):
        """
        Recalculates and sets the new objective expression based on multipliers.
        
        This method constructs a new objective function for the relaxed model by incorporating
        the penalties for violating the hard constraints, scaled by their respective Lagrangian
        multipliers. It also calculates the constant offset added to the objective due to the
        penalization and stores it in the `lagrangian_offset` attribute.
        """
        # Start with a fresh copy of the original objective coefficients
        new_obj_coeffs = {v_name: obj for v_name, obj in self.orig_obj.items()}
        inf = self.model.infinity()
        
        # Manually track the constant offset
        lagrangian_offset = 0.0
        
        # Determine the sign based on optimization sense
        # Minimization: Add the penalty (+1)
        # Maximization: Subtract the penalty (-1)
        sense_sign = 1 if self.orig_sense == "minimize" else -1

        for c_name, c_data in self.hard_conss.items():
            lam = self.multipliers[c_name]
            if lam == 0.0:
                continue
                
            coeffs = c_data["coeffs"]
            rhs = c_data["rhs"]
            lhs = c_data["lhs"]
            
            # For a <= constraint: penalty is lambda * (a^T * x - b)
            if rhs < inf:
                lagrangian_offset -= sense_sign * lam * rhs
                for v_name, coef in coeffs.items():
                    new_obj_coeffs[v_name] += sense_sign * lam * coef
            
            # For a >= constraint: penalty is lambda * (b - a^T * x)
            elif lhs > -inf:
                lagrangian_offset += sense_sign * lam * lhs
                for v_name, coef in coeffs.items():
                    new_obj_coeffs[v_name] -= sense_sign * lam * coef
                    
        # Reconstruct the objective expression using standard PySCIPOpt syntax
        new_obj_expr = 0.0
        var_dict = {v.name: v for v in self.model.getVars()}
        for v_name, new_coef in new_obj_coeffs.items():
            if new_coef != 0.0 and v_name in var_dict:
                new_obj_expr += new_coef * var_dict[v_name]
                
        # Replace the model's objective with our newly built expression
        self.model.setObjective(new_obj_expr, sense=self.orig_sense)
        
        # Store the offset as a class attribute so we can add it to the final solution
        self.lagrangian_offset = lagrangian_offset
        
    def optimize_and_get_violations(self):
        """
        Optimizes the Lagrangian relaxed model and calculates how much 
        the relaxed solution violates the original hard constraints.
        
        Returns:
            tuple: A tuple containing:
                - bound (float or None): The optimal objective value of the relaxed model plus the
                  Lagrangian offset. Returns None if the model could not be solved to optimality.
                - violations (dict or None): A dictionary mapping hard constraint names to their
                  violation amounts (subgradients) in the optimal solution. Returns None if the
                  model could not be solved to optimality.
        """
        self.model.optimize()
        status = self.model.getStatus()
        
        if status not in ("optimal", "gaplimit"):
            print(f"Model did not solve to optimality. Status: {status}")
            return None, None
            
        # 1. Get the Lagrangian bound (use dual bound for valid Lagrangian bound)
        bound = self.model.getDualbound() + getattr(self, "lagrangian_offset", 0.0)
        
        # 2. Calculate the violations (subgradients)
        var_dict = {v.name: v for v in self.model.getVars()}
        violations = {}
        inf = self.model.infinity()
        
        for c_name, c_data in self.hard_conss.items():
            coeffs = c_data["coeffs"]
            rhs = c_data["rhs"]
            lhs = c_data["lhs"]
            
            # Calculate a^T x* (dot product of coefficients and optimal variable values)
            ax_val = sum(coef * self.model.getVal(var_dict[v_name]) 
                         for v_name, coef in coeffs.items() if v_name in var_dict)
            
            violation = 0.0
            
            # Violation for <= constraints: (a^T x* - rhs)
            if rhs < inf:
                violation += (ax_val - rhs)
                
            # Violation for >= constraints: (lhs - a^T x*)
            elif lhs > -inf:
                violation += (lhs - ax_val)
                
            violations[c_name] = violation
            
        return bound, violations