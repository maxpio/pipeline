import time
from pyscipopt import Model as SCIPModel

def get_master_constraints(dec_file: str) -> set:
    """Parses a .dec file and returns a set of constraint names in the MASTERCONSS block."""
    master_conss = set()
    with open(dec_file, 'r') as f:
        in_master = False
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line == "MASTERCONSS":
                in_master = True
                continue
            if line == "MASTERVARS" or line.startswith("BLOCK "):
                in_master = False
                continue
            if in_master:
                master_conss.add(line)
    return master_conss

def extract_features_single(lp_file: str, dec_file: str, maxrounds: int = 0, quiet: bool = False, extract_lp_bound: bool = False) -> dict | tuple[dict, float]:
    """
    Extracts LP relaxation features from a single .lp file using PySCIPOpt.
    If extract_lp_bound is True, returns (feature_dictionary, lp_obj_val).
    Otherwise returns the feature dictionary (same schema as lp_to_json_general.py output).
    
    The dictionary contains:
      - "variables": { var_name: {w, lpr_val, rc_val, is_int}, ... }
      - "constraints": { cons_name: {rhs, eq, dualized, pi}, ... }
      - "edges": [ {c, v, coeff}, ... ]
    """
    if not quiet:
        print(f"[Step 1] Extracting LP relaxation features from: {lp_file}")
    start = time.time()

    master_conss = get_master_constraints(dec_file)

    model = SCIPModel("FeatureExtractor")
    model.hideOutput()
    model.readProblem(str(lp_file))
    model.setIntParam("presolving/maxrounds", maxrounds)

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
            "dualized": 1 if c_name in master_conss else 0
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
                
        lp_obj_val = model.getObjVal()
    else:
        if not quiet:
            print(f"  Warning: LP relaxation status is '{model.getStatus()}', features may be incomplete.")
        # Fill in defaults so the graph can still be built
        for v_name in variables_dict:
            variables_dict[v_name].setdefault("lpr_val", 0.0)
            variables_dict[v_name].setdefault("rc_val", 0.0)
            variables_dict[v_name]["is_int"] = orig_is_int[v_name]
        for c_name in constraints_dict:
            constraints_dict[c_name].setdefault("pi", 0.0)
            
        lp_obj_val = 0.0

    model.freeProb()
    elapsed = time.time() - start
    if not quiet:
        print(f"  Feature extraction done in {elapsed:.2f}s  "
              f"({len(variables_dict)} vars, {len(constraints_dict)} cons, {len(edges_list)} edges)")

    feature_dict = {
        "variables": variables_dict,
        "constraints": constraints_dict,
        "edges": edges_list
    }
    
    if extract_lp_bound:
        return feature_dict, lp_obj_val
    return feature_dict

def generate_feature_json(lp_file: str, dec_file: str, out_json: str, maxrounds: int = 0, quiet: bool = False):
    """
    Generates the .json ML input file for a single instance.
    """
    import json
    feature_dict = extract_features_single(lp_file, dec_file, maxrounds=maxrounds, quiet=quiet)
    with open(out_json, 'w') as f:
        json.dump(feature_dict, f, indent=4)
