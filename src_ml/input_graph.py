import json
import torch
from torch_geometric.data import HeteroData

def create_bipartite_graph(json_path):
    """
    Reads a JSON file containing MIP features and constructs a bipartite 
    HeteroData graph for PyTorch Geometric.
    """
    with open(json_path, 'r') as f:
        data_dict = json.load(f)

    variables = data_dict.get("variables", {})
    constraints = data_dict.get("constraints", {})
    edges = data_dict.get("edges", [])

    data = HeteroData()

    # 1. Process Variable Nodes
    var_name_to_id = {}
    var_features = []
    
    for idx, (v_name, v_feat) in enumerate(variables.items()):
        var_name_to_id[v_name] = idx
        # Order: w, lpr_val, rc_val, is_int
        features = [
            v_feat.get("w", 0.0),
            v_feat.get("lpr_val", 0.0),
            v_feat.get("rc_val", 0.0),
            v_feat.get("is_int", 0.0)
        ]
        var_features.append(features)
        
    # Shape: [num_variables, 4]
    data['variable'].x = torch.tensor(var_features, dtype=torch.float)

    # 2. Process Constraint Nodes
    cons_name_to_id = {}
    cons_features = []
    
    for idx, (c_name, c_feat) in enumerate(constraints.items()):
        cons_name_to_id[c_name] = idx
        # Order: rhs, eq, dualized, pi
        features = [
            c_feat.get("rhs", 0.0),
            c_feat.get("eq", 0.0),
            c_feat.get("dualized", 0.0),
            c_feat.get("pi", 0.0) # Using .get() gracefully handles any missing keys
        ]
        cons_features.append(features)

    # Shape: [num_constraints, 4]
    data['constraint'].x = torch.tensor(cons_features, dtype=torch.float)

    # 3. Process Edges
    src_nodes = [] # Variables
    dst_nodes = [] # Constraints
    edge_attrs = [] # Coefficients
    
    for edge in edges:
        v_name = edge["v"]
        c_name = edge["c"]
        coeff = edge["coeff"]
        
        # Ensure the nodes actually exist in our mapping
        if v_name in var_name_to_id and c_name in cons_name_to_id:
            src_nodes.append(var_name_to_id[v_name])
            dst_nodes.append(cons_name_to_id[c_name])
            edge_attrs.append([coeff]) # PyG expects edge attributes to be 2D: [num_edges, num_edge_features]

    # Convert to Tensors
    edge_index_v2c = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float)

    # Add Forward Edges (Variable -> Constraint)
    data['variable', 'to', 'constraint'].edge_index = edge_index_v2c
    data['variable', 'to', 'constraint'].edge_attr = edge_attr_tensor

    # Add Reverse Edges (Constraint -> Variable)
    # GNNs usually need to pass messages in both directions for bipartite graphs
    edge_index_c2v = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
    data['constraint', 'to', 'variable'].edge_index = edge_index_c2v
    data['constraint', 'to', 'variable'].edge_attr = edge_attr_tensor

    return data
