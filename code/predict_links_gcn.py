import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ----------------
# Model Classes from your training code
# ----------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
    
    def forward(self, x, adj_sparse):
        support = self.linear(x)
        output = torch.sparse.mm(adj_sparse, support)
        return output

class GCNLinkPredictor(nn.Module):
    def __init__(self, in_features, hidden_dim, out_dim, dropout=0.2):
        super(GCNLinkPredictor, self).__init__()
        self.gc1 = GCNLayer(in_features, hidden_dim)
        self.gc2 = GCNLayer(hidden_dim, out_dim)
        self.dropout = dropout
    
    def encode(self, x, adj_sparse):
        x = self.gc1(x, adj_sparse)
        x = F.relu(x)
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj_sparse)
        return x
    
    def decode(self, z, edge_index):
        src, dst = edge_index
        return torch.sum(z[src] * z[dst], dim=1)
    
    def forward(self, x, adj_sparse, edge_index):
        z = self.encode(x, adj_sparse)
        return self.decode(z, edge_index)

# ----------------
# Prediction Functions
# ----------------
def load_model(model_path="models/gcn_link_predictor.pt"):
    """Load the trained model from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model = GCNLinkPredictor(
        in_features=checkpoint['in_features'],
        hidden_dim=checkpoint['hidden_dim'],
        out_dim=checkpoint['out_dim']
    ).to(device)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    return model, checkpoint['in_features']

def process_graph(gml_file_path):
    """Process a GML file and prepare it for prediction."""
    G = nx.read_gml(gml_file_path)
    
    # Save original graph before integer conversion
    original_graph = G.copy()
    
    # Create node mapping from original labels to integers
    node_mapping = {}
    for i, node in enumerate(G.nodes()):
        node_mapping[i] = str(node)  # Store int -> str mapping
    
    # Convert node labels to integers
    G = nx.convert_node_labels_to_integers(G)
    
    # Create adjacency matrix
    adj_dense = nx.to_numpy_array(G)
    adj_tensor = torch.FloatTensor(adj_dense)
    adj_sparse = adj_tensor.to_sparse().coalesce().to(device)
    
    # Create node features (identity matrix)
    num_nodes = len(G.nodes())
    node_features = torch.eye(num_nodes).to(device)
    
    return G, adj_sparse, node_features, original_graph, node_mapping

def generate_all_non_edges(graph, existing_edges):
    """Generate all non-existing edges in the graph."""
    non_edges = []
    edge_set = set()
    
    # Add existing edges to edge_set
    for u, v in existing_edges:
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    # Add all possible non-existing edges
    nodes = list(graph.nodes())
    for i, u in enumerate(nodes):
        for v in nodes[i+1:]:
            if (u, v) not in edge_set:
                non_edges.append((u, v))
    
    return non_edges

def predict_links(model, graph, adj_sparse, node_features, max_features, node_mapping, top_k=None, optimal_threshold=0.5029):
    """Predict links for the input graph using optimal threshold."""
    # Pad features if necessary
    if node_features.shape[1] < max_features:
        padding = torch.zeros(node_features.shape[0], max_features - node_features.shape[1]).to(device)
        node_features = torch.cat([node_features, padding], dim=1)
    
    # Get existing edges
    existing_edges = list(graph.edges())
    
    # Generate all non-existing edges
    non_edges = generate_all_non_edges(graph, existing_edges)
    print(f"Analyzing {len(non_edges)} potential links...")
    
    # Predict scores for non-existing edges
    with torch.no_grad():
        z = model.encode(node_features, adj_sparse)
        
        # Process in batches to avoid memory issues
        batch_size = 1000
        all_scores = []
        all_pairs = []
        
        for i in range(0, len(non_edges), batch_size):
            batch_edges = non_edges[i:i+batch_size]
            edge_tensor = torch.tensor(batch_edges, dtype=torch.long).t().to(device)
            scores = model.decode(z, edge_tensor).cpu().numpy()
            
            all_scores.extend(scores)
            all_pairs.extend(batch_edges)
        
        all_scores = np.array(all_scores)
        
        # Apply sigmoid to get probabilities
        probabilities = 1 / (1 + np.exp(-all_scores))
        
        # Apply optimal threshold
        predicted_edges = []
        for i, prob in enumerate(probabilities):
            if prob >= optimal_threshold:
                u, v = all_pairs[i]
                orig_u = node_mapping[u]
                orig_v = node_mapping[v]
                predicted_edges.append((orig_u, orig_v, prob))
        
        # Sort by probability
        predicted_edges.sort(key=lambda x: x[2], reverse=True)
        
        if top_k is not None:
            predicted_edges = predicted_edges[:top_k]
    
    return predicted_edges

def create_enhanced_graph(original_graph, predictions):
    """Create a new graph with predicted edges added."""
    enhanced_graph = original_graph.copy()
    
    # Add predicted edges with probability as an edge attribute
    for u, v, prob in predictions:
        enhanced_graph.add_edge(u, v, predicted=True, probability=float(prob))
    
    return enhanced_graph

def main(gml_file_path, model_path="models/gcn_link_predictor.pt", top_k=None, optimal_threshold=0.5029):
    """Main function to predict links for a given GML file and create enhanced graph."""
    # Load the model
    print("Loading trained model...")
    model, max_features = load_model(model_path)
    
    # Process the graph
    print(f"Processing graph from {gml_file_path}...")
    graph, adj_sparse, node_features, original_graph, node_mapping = process_graph(gml_file_path)
    print(f"Graph has {len(graph.nodes())} nodes and {len(graph.edges())} edges")
    
    # Predict links using optimal threshold
    print(f"Predicting links using threshold: {optimal_threshold}")
    predictions = predict_links(model, graph, adj_sparse, node_features, max_features, node_mapping, top_k, optimal_threshold)
    
    # Create enhanced graph with predicted edges
    enhanced_graph = create_enhanced_graph(original_graph, predictions)
    
    # Save enhanced graph to new GML file
    output_file = os.path.splitext(gml_file_path)[0] + "_with_predictions_gcn.gml"
    nx.write_gml(enhanced_graph, output_file)
    print(f"\nEnhanced graph saved to {output_file}")
    
    # Also save predictions to text file
    text_output_file = os.path.splitext(gml_file_path)[0] + "_predictions_gcn.txt"
    with open(text_output_file, 'w') as f:
        f.write(f"Link predictions for {gml_file_path}\n")
        f.write(f"Generated using model: {model_path}\n")
        f.write(f"Using optimal threshold: {optimal_threshold}\n")
        f.write(f"Number of predicted edges: {len(predictions)}\n")
        f.write("-" * 50 + "\n")
        for i, (u, v, prob) in enumerate(predictions, 1):
            f.write(f"{i}. {u} <-> {v}: {prob:.4f}\n")
    
    print(f"Predictions text file saved to {text_output_file}")
    
    # Print summary
    print(f"\nPredicted and added {len(predictions)} new edges to the graph")
    print(f"Original graph had {len(original_graph.edges())} edges")
    print(f"Enhanced graph has {len(enhanced_graph.edges())} edges")
    
    return enhanced_graph, predictions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Predict links in a GML file using trained GCN model and create enhanced graph")
    parser.add_argument("gml_file", help="Path to the input GML file")
    parser.add_argument("--model", default="models/gcn_link_predictor.pt", help="Path to the trained model file")
    parser.add_argument("--top_k", type=int, default=None, help="Number of top predictions to add (default: all above threshold)")
    parser.add_argument("--threshold", type=float, default=0.5029, help="Probability threshold (default: 0.5029 - optimal from evaluation)")
    
    args = parser.parse_args()
    
    main(args.gml_file, args.model, args.top_k, args.threshold)