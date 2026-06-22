"""
Constructs a bipartite HeteroData graph for PyTorch Geometric from MIP features.
"""
import json
import torch
from torch_geometric.data import HeteroData

def create_bipartite_graph(json_path):
    """Builds a bipartite HeteroData graph from a JSON file."""
    with open(json_path, 'r') as f:
        data_dict = json.load(f)
    return create_bipartite_graph_from_dict(data_dict)

def create_bipartite_graph_from_dict(data_dict):
    """Builds a bipartite HeteroData graph from a feature dictionary."""
    variables = data_dict.get("variables", {})
    constraints = data_dict.get("constraints", {})
    edges = data_dict.get("edges", [])

    data = HeteroData()

    # Process Variables
    var_name_to_id = {}
    var_features = []
    
    for idx, (v_name, v_feat) in enumerate(variables.items()):
        var_name_to_id[v_name] = idx
        # Set features
        features = [
            v_feat.get("w", 0.0),
            v_feat.get("lpr_val", 0.0),
            v_feat.get("rc_val", 0.0),
            v_feat.get("is_int", 0.0)
        ]
        var_features.append(features)
        
    # Build tensor
    data['variable'].x = torch.tensor(var_features, dtype=torch.float)

    # Process Constraints
    cons_name_to_id = {}
    cons_features = []
    
    for idx, (c_name, c_feat) in enumerate(constraints.items()):
        cons_name_to_id[c_name] = idx
        # Set features
        features = [
            c_feat.get("rhs", 0.0),
            c_feat.get("eq", 0.0),
            c_feat.get("dualized", 0.0),
            c_feat.get("pi", 0.0)
        ]
        cons_features.append(features)

    # Build tensor
    data['constraint'].x = torch.tensor(cons_features, dtype=torch.float)

    # Process Edges
    src_nodes = []
    dst_nodes = []
    edge_attrs = []
    
    for edge in edges:
        v_name = edge["v"]
        c_name = edge["c"]
        coeff = edge["coeff"]
        
        # Check nodes exist
        if v_name in var_name_to_id and c_name in cons_name_to_id:
            src_nodes.append(var_name_to_id[v_name])
            dst_nodes.append(cons_name_to_id[c_name])
            edge_attrs.append([coeff])

    # Create Tensors
    edge_index_v2c = torch.tensor([src_nodes, dst_nodes], dtype=torch.long)
    edge_attr_tensor = torch.tensor(edge_attrs, dtype=torch.float)

    # Forward Edges
    data['variable', 'to', 'constraint'].edge_index = edge_index_v2c
    data['variable', 'to', 'constraint'].edge_attr = edge_attr_tensor

    # Reverse Edges
    edge_index_c2v = torch.tensor([dst_nodes, src_nodes], dtype=torch.long)
    data['constraint', 'to', 'variable'].edge_index = edge_index_c2v
    data['constraint', 'to', 'variable'].edge_attr = edge_attr_tensor

    return data
