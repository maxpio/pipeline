"""
Defines the PyTorch model for predicting Lagrangian multipliers using a bipartite GNN.
"""
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing

class EncoderMLP(nn.Module):
    """A standard MLP block."""
    def __init__(self, layer_dims):
        """Initializes the MLP layers."""
        super().__init__()
        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i+1]))
            if i < len(layer_dims) - 2:
                layers.append(nn.ReLU())
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """Passes input through the MLP."""
        return self.encoder(x)

class InitialNodeEncoders(nn.Module):
    """Encodes raw features into a shared hidden dimension."""
    def __init__(self, encoder_mlp_dims=[64]):
        """Initializes encoders."""
        super().__init__()
        
        # Var encoder
        self.var_encoder = EncoderMLP([4] + encoder_mlp_dims)
        
        # Cons encoder
        self.cons_encoder = EncoderMLP([4] + encoder_mlp_dims)

    def forward(self, x_dict):
        """Encodes dictionary of node features."""
        var_x = x_dict['variable']
        cons_x = x_dict['constraint']
        
        var_h = self.var_encoder(var_x)
        cons_h = self.cons_encoder(cons_x)
        
        # Return embeddings
        return {
            'variable': var_h,
            'constraint': cons_h
        }
    
class BipartiteUpdate(MessagePassing):
    """Generic message passing layer for bipartite graphs."""
    def __init__(self, feature_embedding_size, gnn_mlp_dims, dropout=0.1):
        """Initializes message passing components."""
        # Add aggregation
        super().__init__(aggr='add') 
        
        # Learnable weights
        self.W = nn.Linear(feature_embedding_size, feature_embedding_size, bias=False)
        
        # Update MLP
        layers = []
        for i in range(len(gnn_mlp_dims) - 1):
            layers.append(nn.Linear(gnn_mlp_dims[i], gnn_mlp_dims[i+1]))
            if i < len(gnn_mlp_dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout))
                
        self.mlp = nn.Sequential(*layers)
        
        # Layer norm
        self.norm = nn.LayerNorm(feature_embedding_size)

    def forward(self, source_x, target_x, edge_index, edge_attr):
        """Propagates and updates node features."""
        # Transform source
        transformed_source = self.W(source_x)
        
        # Propagate messages
        m = self.propagate(edge_index, x=(transformed_source, target_x), edge_attr=edge_attr)
        
        # Concat features
        concat_features = torch.cat([target_x, m], dim=-1)
        
        # Apply MLP
        update = self.mlp(concat_features)
        
        # Add residual
        new_target_x = target_x + update
        
        # Apply norm
        new_target_x = self.norm(new_target_x)
        
        return new_target_x

    def message(self, x_j, edge_attr):
        """Constructs messages using edge attributes."""
        # Scale message
        return x_j * edge_attr.view(-1, 1) 


class BipartiteGNNBlock(nn.Module):
    """Executes one full block of bipartite message passing."""
    def __init__(self, feature_embedding_size, gnn_mlp_dims, dropout=0.1):
        """Initializes bidirectional update layers."""
        super().__init__()
        self.cons_to_var = BipartiteUpdate(feature_embedding_size, gnn_mlp_dims, dropout=dropout)
        self.var_to_cons = BipartiteUpdate(feature_embedding_size, gnn_mlp_dims, dropout=dropout)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """Performs two-step message passing."""
        var_x = x_dict['variable']
        cons_x = x_dict['constraint']

        # Step A
        edge_index_c2v = edge_index_dict[('constraint', 'to', 'variable')]
        edge_attr_c2v = edge_attr_dict[('constraint', 'to', 'variable')]
        
        updated_var_x = self.cons_to_var(
            source_x=cons_x, 
            target_x=var_x, 
            edge_index=edge_index_c2v, 
            edge_attr=edge_attr_c2v
        )
        
        # Step B
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
    """Stacks multiple BipartiteGNNBlock layers."""
    def __init__(self, feature_embedding_size, gnn_mlp_dims, num_message_passing_layers, dropout=0.1):
        """Initializes the stacked GNN layers."""
        super().__init__()
        self.layers = nn.ModuleList([
            BipartiteGNNBlock(feature_embedding_size, gnn_mlp_dims, dropout=dropout) for _ in range(num_message_passing_layers)
        ])

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """Passes data through all GNN layers."""
        current_x_dict = x_dict
        
        for layer in self.layers:
            current_x_dict = layer(current_x_dict, edge_index_dict, edge_attr_dict)
            
        return current_x_dict
class Decoder(nn.Module):
    """Predicts final Lagrangian multipliers from relaxed constraints."""
    def __init__(self, feature_embedding_size, decoder_mlp_dims):
        """Initializes decoder MLP."""
        super().__init__()
        # Build MLP dims
        full_mlp_dims = [feature_embedding_size] + decoder_mlp_dims + [1]
        # Init MLP
        self.mlp = EncoderMLP(full_mlp_dims)

    def forward(self, cons_h, raw_cons_features):
        """Decodes hidden states into multiplier predictions."""
        # Extract features
        eq_flags = raw_cons_features[:, 1]
        dualized_flags = raw_cons_features[:, 2]
        pi_vals = raw_cons_features[:, 3]

        # Filter relaxed constraints
        mask = dualized_flags > 0.5

        filtered_h = cons_h[mask]
        filtered_pi = pi_vals[mask]
        filtered_eq = eq_flags[mask]

        # Predict residual
        delta_pi = self.mlp(filtered_h).squeeze(-1) 

        # Final multiplier
        lambda_raw = filtered_pi + delta_pi

        return lambda_raw, mask, filtered_eq


class LagrangianMultiplierModel(nn.Module):
    """Complete model pipeline: Encoders, GNN, Decoder."""
    def __init__(self, feature_embedding_size=64, encoder_mlp_dims=[4, 64], gnn_mlp_dims=[64], num_message_passing_layers=3, dropout=0.1, decoder_mlp_dims=[]):
        """Initializes all model components."""
        super().__init__()
        
        # Build dims
        full_encoder_dims = encoder_mlp_dims + [feature_embedding_size]
        full_gnn_dims = [feature_embedding_size * 2] + gnn_mlp_dims + [feature_embedding_size]
            
        # Init encoders
        self.encoders = InitialNodeEncoders(encoder_mlp_dims=full_encoder_dims)
        
        # Init GNN
        self.gnn = StackedBipartiteGNN(
            feature_embedding_size=feature_embedding_size, gnn_mlp_dims=full_gnn_dims, num_message_passing_layers=num_message_passing_layers, dropout=dropout
        )
        
        # Init decoder
        self.decoder = Decoder(feature_embedding_size=feature_embedding_size, decoder_mlp_dims=decoder_mlp_dims)

    def forward(self, data):
        """Executes full forward pass on PyG HeteroData."""
        # Save raw features
        raw_cons_features = data['constraint'].x

        # Encode
        x_dict = self.encoders(data.x_dict)

        # Pass messages
        x_dict = self.gnn(x_dict, data.edge_index_dict, data.edge_attr_dict)
    
        # Decode
        lambda_raw, dualized_mask, eq_flags = self.decoder(x_dict['constraint'], raw_cons_features)

        return lambda_raw, dualized_mask, eq_flags