import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EncoderMLP(nn.Module):
    """
    A standard MLP block for the initial node encoders.
    Sequence: Linear -> (ReLU -> Linear)*
    """
    def __init__(self, layer_dims):
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.encoder(x)

class InitialNodeEncoders(nn.Module):
    """
    Processes the raw features for variables and constraints into a 
    shared hidden dimension prior to message passing.
    """
    def __init__(self, encoder_mlp_dims=[64]):
        super().__init__()
        
        
        # Variable Encoder (3 continuous + 1 binary)
        self.var_encoder = EncoderMLP([4] + encoder_mlp_dims)
        
        # Constraint Encoder (2 continuous + 2 binary)
        self.cons_encoder = EncoderMLP([4] + encoder_mlp_dims)

    def forward(self, x_dict):
        """
        Expects a dictionary of node features, typical when working with 
        PyG's HeteroData (e.g., hetero_data.x_dict).
        """
        var_x = x_dict['variable']
        cons_x = x_dict['constraint']
        
        var_h = self.var_encoder(var_x)
        
        cons_h = self.cons_encoder(cons_x)
        
        # Return an updated dictionary of embedded node features
        return {
            'variable': var_h,
            'constraint': cons_h
        }
    
class BipartiteUpdate(MessagePassing):
    """
    A generic message passing layer that can handle both:
    - Step A: Constraints to Variables
    - Step B: Variables to Constraints
    """
    def __init__(self, feature_embedding_size, gnn_mlp_dims, dropout=0.1):
        # aggr='add' corresponds to the summation in your formulas
        super().__init__(aggr='add') 
        
        # The learnable weight matrix W (W_C for Step A, W_V for Step B)
        self.W = nn.Linear(feature_embedding_size, feature_embedding_size, bias=False)
        
        # The MLP applied to the concatenation of the current state and the message
        layers = []
        for i in range(len(gnn_mlp_dims) - 1):
            layers.append(nn.Linear(gnn_mlp_dims[i], gnn_mlp_dims[i+1]))
            if i < len(gnn_mlp_dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
                
        self.mlp = nn.Sequential(*layers)
        
        # LayerNorm layer
        self.norm = nn.LayerNorm(feature_embedding_size)

    def forward(self, source_x, target_x, edge_index, edge_attr):
        """
        source_x: Features of the nodes sending messages (e.g., h_c)
        target_x: Features of the nodes receiving messages (e.g., h_v)
        """
        # Apply the learnable weight matrix to the source features: W * h^(l)
        transformed_source = self.W(source_x)
        
        # Propagate messages. We explicitly pass x as a tuple of (source, target)
        # to handle the bipartite nature correctly.
        m = self.propagate(edge_index, x=(transformed_source, target_x), edge_attr=edge_attr)
        
        # Update Step: Concatenate current state and aggregated message (h || m)
        concat_features = torch.cat([target_x, m], dim=-1)
        
        # 1. Apply MLP
        update = self.mlp(concat_features)
        
        # 2. Add residual connection
        new_target_x = target_x + update
        
        # 3. Apply LayerNorm (NEW)
        new_target_x = self.norm(new_target_x)
        
        return new_target_x

    def message(self, x_j, edge_attr):
        """
        x_j: The source node features that have already been transformed by W.
        edge_attr: The edge coefficients A_{vc}.
        """
        # Ensure edge_attr broadcasts correctly across the hidden dimensions
        # Scales the transformed source features by the raw edge coefficient
        return x_j * edge_attr.view(-1, 1) 


class BipartiteGNNBlock(nn.Module):
    """
    Executes one full block of message passing sequentially:
    Step A: Constraints -> Variables
    Step B: Variables -> Constraints
    """
    def __init__(self, feature_embedding_size, gnn_mlp_dims, dropout=0.1):
        super().__init__()
        self.cons_to_var = BipartiteUpdate(feature_embedding_size, gnn_mlp_dims, dropout=dropout) # Step A
        self.var_to_cons = BipartiteUpdate(feature_embedding_size, gnn_mlp_dims, dropout=dropout) # Step B

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        var_x = x_dict['variable']
        cons_x = x_dict['constraint']

        # Step A: Constraints to Variables
        edge_index_c2v = edge_index_dict[('constraint', 'to', 'variable')]
        edge_attr_c2v = edge_attr_dict[('constraint', 'to', 'variable')]
        
        updated_var_x = self.cons_to_var(
            source_x=cons_x, 
            target_x=var_x, 
            edge_index=edge_index_c2v, 
            edge_attr=edge_attr_c2v
        )
        
        # Step B: Variables to Constraints
        # Note: We use the *updated* variable features here, matching h_v^{(l+1)} in the formula
        edge_index_v2c = edge_index_dict[('variable', 'to', 'constraint')]
        edge_attr_v2c = edge_attr_dict[('variable', 'to', 'constraint')]
        
        updated_cons_x = self.var_to_cons(
            source_x=updated_var_x, 
            target_x=cons_x, 
            edge_index=edge_index_v2c, 
            edge_attr=edge_attr_v2c
        )

        return {'variable': updated_var_x, 'constraint': updated_cons_x}


class StackedBipartiteGNN(nn.Module):
    """
    Stacks L layers of the BipartiteGNNBlock.
    """
    def __init__(self, feature_embedding_size, gnn_mlp_dims, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            BipartiteGNNBlock(feature_embedding_size, gnn_mlp_dims, dropout=dropout) for _ in range(num_layers)
        ])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        current_x_dict = x_dict
        
        for layer in self.layers:
            current_x_dict = layer(current_x_dict, edge_index_dict, edge_attr_dict)
            
        return current_x_dict
class Decoder(nn.Module):
    """
    Isolates the relaxed constraints and predicts the final Lagrangian multipliers.
    """
    def __init__(self, feature_embedding_size, decoder_mlp_dims):
        super().__init__()
        # Construct the full layer dimensions for the MLP:
        # Input: feature_embedding_size
        # Hidden layers: decoder_mlp_dims
        # Output: 1
        full_mlp_dims = [feature_embedding_size] + decoder_mlp_dims + [1]
        # Reuse EncoderMLP as it implements the desired Linear-ReLU sequence for hidden layers
        # and a final Linear layer without ReLU.
        self.mlp = EncoderMLP(full_mlp_dims)

    def forward(self, cons_h, raw_cons_features):
        """
        cons_h: The hidden embeddings of the constraints after L message passing layers.
        raw_cons_features: The original constraint features [rhs, eq, dualized, pi].
        """
        # Feature indices based on your initial extraction order:
        # 0: rhs, 1: eq, 2: dualized, 3: pi
        eq_flags = raw_cons_features[:, 1]
        dualized_flags = raw_cons_features[:, 2]
        pi_vals = raw_cons_features[:, 3]

        # 1. Filtering: Create a boolean mask for constraints where dualized == 1.0
        # This implicitly drops all variables (since we only pass cons_h) and non-relaxed constraints
        mask = dualized_flags > 0.5

        filtered_h = cons_h[mask]
        filtered_pi = pi_vals[mask]
        filtered_eq = eq_flags[mask]

        # 2. Residual Prediction (Delta pi)
        delta_pi = self.mlp(filtered_h).squeeze(-1) 

        # 3. Final Multiplier (Raw)
        lambda_raw = filtered_pi + delta_pi

        return lambda_raw, mask, filtered_eq


class LagrangianMultiplierModel(nn.Module):
    """
    The complete model pipeline combining Encoders, Message Passing, and the Decoder.
    """
    def __init__(self, feature_embedding_size=64, encoder_mlp_dims=[4, 64], gnn_mlp_dims=[64], num_layers=3, dropout=0.1, decoder_mlp_dims=[]):
        super().__init__()
        
        # Dynamically append dimensions that depend on feature_embedding_size
        full_encoder_dims = encoder_mlp_dims + [feature_embedding_size]
        full_gnn_dims = [feature_embedding_size * 2] + gnn_mlp_dims + [feature_embedding_size]
            
        # Step 1: Encoders
        self.encoders = InitialNodeEncoders(encoder_mlp_dims=full_encoder_dims)
        
        # Step 2: Message Passing Blocks
        self.gnn = StackedBipartiteGNN(
            feature_embedding_size=feature_embedding_size, gnn_mlp_dims=full_gnn_dims, num_layers=num_layers, dropout=dropout
        )
        
        # Step 3: Decoder
        self.decoder = Decoder(feature_embedding_size=feature_embedding_size, decoder_mlp_dims=decoder_mlp_dims)

    def forward(self, data):
        """
        Takes the full PyG HeteroData object as input.
        """
        # Keep a reference to the original constraint features for the decoder
        raw_cons_features = data['constraint'].x

        # 1. Encode
        x_dict = self.encoders(data.x_dict)

        # 2. Pass Messages
        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_attr_dict)
    
        # 3. Decode
        lambda_raw, dualized_mask, eq_flags = self.decoder(x_dict['constraint'], raw_cons_features)

        return lambda_raw, dualized_mask, eq_flags