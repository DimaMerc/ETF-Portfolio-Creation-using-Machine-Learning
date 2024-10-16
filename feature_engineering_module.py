# feature_engineering_module.py



import numpy as np
import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def create_graph_data(etf_holdings, equity_data):
    node_features = []
    edge_index = []
    edge_weights = []
    node_mapping = {}
    current_index = 0
    etf_node_indices = []

    NUM_NODE_FEATURES = 10  # Assuming we use RSI and MACD as features

    # Create nodes for ETFs
    for etf_symbol in etf_holdings.keys():
        # For ETFs, we can use placeholder features or specific ETF features if available
        features = [0.0] * NUM_NODE_FEATURES  # Placeholder features
        node_features.append(features)
        node_mapping[etf_symbol] = current_index
        etf_node_indices.append(current_index)
        current_index += 1

    # Create nodes for equities
    for symbol in equity_data.keys():
        df = equity_data[symbol]
        # Extract the latest values of the technical indicators
        features = df.iloc[-1][[
            'RSI_14', 'MACD', 'SMA_20', 'EMA_20',
            'BBANDS_upper', 'BBANDS_middle', 'BBANDS_lower',
            'ADX_14', 'ROC_10', 'ATR_14'
        ]].values.tolist()
        # Ensure that features have the correct length
        if len(features) == NUM_NODE_FEATURES:
            node_features.append(features)
            node_mapping[symbol] = current_index
            current_index += 1
        else:
            # Handle the case where features are missing
            print(f"Missing features for {symbol}")
            # decide to skip this node or fill missing values

    # Create edges between ETFs and their holdings
    for etf_symbol, holdings_df in etf_holdings.items():
        etf_idx = node_mapping[etf_symbol]
        for _, row in holdings_df.iterrows():
            equity_symbol = row['symbol']
            if equity_symbol in node_mapping:
                equity_idx = node_mapping[equity_symbol]
                weight = float(row['weight']) / 100.0  # Convert percentage to fraction between 0 and 1
                # Add edges in both directions with weights
                edge_index.append([etf_idx, equity_idx])
                edge_weights.append(weight)
                edge_index.append([equity_idx, etf_idx])
                edge_weights.append(weight)

    # Convert to tensors
    x = torch.tensor(node_features, dtype=torch.float)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weights, dtype=torch.float)
    etf_mask = torch.zeros(x.size(0), dtype=torch.bool)
    etf_mask[etf_node_indices] = True

    # Create data object
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    data.etf_mask = etf_mask
    data.num_node_features = x.shape[1]

    return data, node_mapping

def generate_gcn_features(data):
    

    class SimpleGCN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels=32):
            super(SimpleGCN, self).__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, out_channels)

        def forward(self, data):
            x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight
            x = self.conv1(x, edge_index, edge_weight=edge_weight)
            x = F.relu(x)
            x = self.conv2(x, edge_index, edge_weight=edge_weight)
            return x

    gcn_model = SimpleGCN(
        in_channels=data.num_node_features,
        hidden_channels=16,
        out_channels=32
    )

    # For simplicity, you can initialize the GCN model without training
    # Ideally, you should train the GCN model here if you have the appropriate data
    #find weights from somwhere on hugging face??!!!!!

    # Generate embeddings
    gcn_model.eval()
    with torch.no_grad():
        embeddings = gcn_model(data)

    return embeddings
