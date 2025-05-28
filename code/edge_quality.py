import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import pandas as pd
import seaborn as sns

class EdgeQualityEvaluator:
    def __init__(self, original_graph, enhanced_graph):
        self.original_graph = original_graph
        self.enhanced_graph = enhanced_graph
        
        # Separate predicted edges from original edges
        self.predicted_edges = []
        for u, v, data in enhanced_graph.edges(data=True):
            if 'predicted' in data and data['predicted']:
                self.predicted_edges.append((u, v, data['probability']))
    
    def jaccard_coefficient(self, u, v):
        """Calculate Jaccard coefficient for a node pair."""
        neighbors_u = set(self.original_graph.neighbors(u))
        neighbors_v = set(self.original_graph.neighbors(v))
        
        if len(neighbors_u.union(neighbors_v)) == 0:
            return 0.0
        
        return len(neighbors_u.intersection(neighbors_v)) / len(neighbors_u.union(neighbors_v))
    
    def adamic_adar_index(self, u, v):
        """Calculate Adamic-Adar index for a node pair."""
        neighbors_u = set(self.original_graph.neighbors(u))
        neighbors_v = set(self.original_graph.neighbors(v))
        common_neighbors = neighbors_u.intersection(neighbors_v)
        
        index = 0
        for w in common_neighbors:
            degree_w = self.original_graph.degree(w)
            if degree_w > 1:
                index += 1 / np.log(degree_w)
        
        return index
    
    def common_neighbors(self, u, v):
        """Count common neighbors between nodes."""
        neighbors_u = set(self.original_graph.neighbors(u))
        neighbors_v = set(self.original_graph.neighbors(v))
        return len(neighbors_u.intersection(neighbors_v))
    
    def preferential_attachment(self, u, v):
        """Calculate preferential attachment score."""
        return self.original_graph.degree(u) * self.original_graph.degree(v)
    
    def resource_allocation(self, u, v):
        """Calculate resource allocation index."""
        neighbors_u = set(self.original_graph.neighbors(u))
        neighbors_v = set(self.original_graph.neighbors(v))
        common_neighbors = neighbors_u.intersection(neighbors_v)
        
        index = 0
        for w in common_neighbors:
            degree_w = self.original_graph.degree(w)
            if degree_w > 0:
                index += 1 / degree_w
        
        return index
    
    def edge_betweenness_impact(self, u, v):
        """Measure how adding this edge affects path lengths in the graph."""
        # Create copy without the edge
        temp_graph = self.original_graph.copy()
        
        # Calculate shortest paths before adding edge (if not already connected)
        if not temp_graph.has_edge(u, v):
            try:
                original_path_length = nx.shortest_path_length(temp_graph, u, v)
            except:
                original_path_length = float('inf')
            
            # Add edge and recalculate
            temp_graph.add_edge(u, v)
            
            # Calculate number of shortest paths affected
            affected_paths = 0
            for source in temp_graph.nodes():
                for target in temp_graph.nodes():
                    if source < target:  # Avoid double counting
                        try:
                            # Check if new edge is used in shortest path
                            path = nx.shortest_path(temp_graph, source, target)
                            if (u, v) in zip(path[:-1], path[1:]) or (v, u) in zip(path[:-1], path[1:]):
                                affected_paths += 1
                        except:
                            continue
            
            return affected_paths, original_path_length - 1
        else:
            return 0, 0
    
    def evaluate_predictions(self):
        """Evaluate the quality of all predicted edges."""
        results = []
        
        for u, v, prob in self.predicted_edges:
            # Skip if nodes don't exist in original graph
            if u not in self.original_graph or v not in self.original_graph:
                continue
            
            # Calculate all metrics
            jaccard = self.jaccard_coefficient(u, v)
            adamic_adar = self.adamic_adar_index(u, v)
            common_neighbors_count = self.common_neighbors(u, v)
            pref_attach = self.preferential_attachment(u, v)
            resource_alloc = self.resource_allocation(u, v)
            affected_paths, path_reduction = self.edge_betweenness_impact(u, v)
            
            # Store results
            results.append({
                'source': u,
                'target': v,
                'probability': prob,
                'jaccard_coefficient': jaccard,
                'adamic_adar': adamic_adar,
                'common_neighbors': common_neighbors_count,
                'preferential_attachment': pref_attach,
                'resource_allocation': resource_alloc,
                'affected_paths': affected_paths,
                'path_reduction': path_reduction
            })
        
        # Create DataFrame for analysis
        df = pd.DataFrame(results)
        
        # Add quality score based on multiple metrics
        if not df.empty:
            df['quality_score'] = (
                df['jaccard_coefficient'].rank(pct=True) * 0.25 +
                df['adamic_adar'].rank(pct=True) * 0.25 +
                df['common_neighbors'].rank(pct=True) * 0.2 +
                df['resource_allocation'].rank(pct=True) * 0.2 +
                df['affected_paths'].rank(pct=True) * 0.1
            )
        
        return df
    
    def visualize_quality_distributions(self, df):
        """Visualize the distribution of quality metrics."""
        metrics = ['jaccard_coefficient', 'adamic_adar', 'common_neighbors', 
                  'resource_allocation', 'quality_score']
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            sns.histplot(data=df, x=metric, ax=axes[i])
            axes[i].set_title(f'Distribution of {metric}')
        
        # Also plot probability vs quality_score
        sns.scatterplot(data=df, x='probability', y='quality_score', ax=axes[5])
        axes[5].set_title('Model Probability vs Quality Score')
        
        plt.tight_layout()
        plt.savefig('edge_quality_distributions.png')
        plt.close()
        
        # Create correlation matrix
        correlation_matrix = df[metrics + ['probability']].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation between Quality Metrics and Prediction Probability')
        plt.tight_layout()
        plt.savefig('edge_quality_correlations.png')
        plt.close()
    
    def analyze_local_structure(self):
        """Analyze how predictions affect local graph structure."""
        metrics = defaultdict(dict)
        
        # Get nodes involved in predictions
        predicted_nodes = set()
        for u, v, _ in self.predicted_edges:
            predicted_nodes.add(u)
            predicted_nodes.add(v)
        
        for node in predicted_nodes:
            # Original metrics
            metrics[node]['original_degree'] = self.original_graph.degree(node)
            metrics[node]['original_clustering'] = nx.clustering(self.original_graph, node)
            
            # Enhanced metrics
            metrics[node]['enhanced_degree'] = self.enhanced_graph.degree(node)
            metrics[node]['enhanced_clustering'] = nx.clustering(self.enhanced_graph, node)
            
            # Change metrics
            metrics[node]['degree_change'] = metrics[node]['enhanced_degree'] - metrics[node]['original_degree']
            metrics[node]['clustering_change'] = metrics[node]['enhanced_clustering'] - metrics[node]['original_clustering']
        
        return pd.DataFrame.from_dict(metrics, orient='index')
    
    def identify_best_worst_predictions(self, df, n=10):
        """Identify the best and worst predictions based on quality score."""
        print("\nTop {} Best Predictions:".format(n))
        best = df.nlargest(n, 'quality_score')
        print(best[['source', 'target', 'probability', 'quality_score', 'common_neighbors', 'jaccard_coefficient']])
        
        print("\nTop {} Worst Predictions:".format(n))
        worst = df.nsmallest(n, 'quality_score')
        print(worst[['source', 'target', 'probability', 'quality_score', 'common_neighbors', 'jaccard_coefficient']])
        
        return best, worst

def main(original_gml_path, enhanced_gml_path, output_dir="quality_analysis"):
    """Main function to evaluate predicted edge quality."""
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load graphs
    print("Loading graphs...")
    original_graph = nx.read_gml(original_gml_path)
    enhanced_graph = nx.read_gml(enhanced_gml_path)
    
    # Initialize evaluator
    evaluator = EdgeQualityEvaluator(original_graph, enhanced_graph)
    
    # Evaluate predictions
    print("Evaluating predicted edges...")
    results_df = evaluator.evaluate_predictions()
    
    # Save detailed results
    results_df.to_csv(os.path.join(output_dir, 'edge_quality_metrics.csv'), index=False)
    
    # Visualize quality distributions
    print("Creating visualizations...")
    evaluator.visualize_quality_distributions(results_df)
    
    # Analyze local structure
    structure_df = evaluator.analyze_local_structure()
    structure_df.to_csv(os.path.join(output_dir, 'local_structure_analysis.csv'))
    
    # Identify best/worst predictions
    best, worst = evaluator.identify_best_worst_predictions(results_df)
    
    # Generate summary report
    with open(os.path.join(output_dir, 'quality_report.txt'), 'w') as f:
        f.write("Edge Prediction Quality Analysis Report\n")
        f.write("=" * 40 + "\n\n")
        
        f.write(f"Total predicted edges: {len(evaluator.predicted_edges)}\n")
        f.write(f"Original graph edges: {len(original_graph.edges())}\n")
        f.write(f"Enhanced graph edges: {len(enhanced_graph.edges())}\n\n")
        
        f.write("Overall Quality Statistics:\n")
        f.write(results_df.describe().to_string())
        f.write("\n\n")
        
        f.write("Correlation with Model Probability:\n")
        correlations = results_df.corr()['probability'].sort_values(ascending=False)
        f.write(correlations.to_string())
        f.write("\n\n")
        
        f.write("Average Quality Metrics by Probability Quartile:\n")
        results_df['prob_quartile'] = pd.qcut(results_df['probability'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
        quartile_stats = results_df.groupby('prob_quartile').mean()
        f.write(quartile_stats.to_string())
    
    print(f"Analysis complete. Results saved to {output_dir}/")
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate the quality of predicted edges")
    parser.add_argument("original_gml", help="Path to the original GML file")
    parser.add_argument("enhanced_gml", help="Path to the enhanced GML file with predictions")
    parser.add_argument("--output_dir", default="quality_analysis", help="Directory to save results")
    
    args = parser.parse_args()
    main(args.original_gml, args.enhanced_gml, args.output_dir)