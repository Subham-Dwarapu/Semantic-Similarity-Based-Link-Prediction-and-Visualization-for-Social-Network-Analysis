#  Semantic Similarity-Based Link Prediction and Visualization for Social Network Analysis

This project aims to predict missing or potential links in a social network using deep learning and graph embedding techniques. The primary focus is on visualizing semantic similarities between nodes using models like GCN, GAT, and GraphSAGE to enhance the interpretability and effectiveness of link prediction.

---

# Problem Statement

Social networks are complex and often contain hidden or evolving relationships. This project addresses:
- Prediction of missing or future connections using **spectral and Node2Vec embeddings**
- Application of **Graph Neural Networks (GNNs)** like **GCN**, **GAT**, and **GraphSAGE**
- Visualization of semantic similarity using interactive dashboards

---

# Dataset

- **Source**: [Noesis Link Prediction Datasets](https://noesis.ikor.org/datasets/link-prediction)
- **Format**: Pajek `.net` graph files
- **Nodes**: ~5,155  
- **Edges**: ~39,285  
- **Graph Type**: Directed

---

# Methodology

1. **Preprocessing**
   - Clean and format graph data
   - Remove self-loops and isolated nodes

2. **Embedding Generation**
   - For small graphs: Spectral Embedding
   - For large graphs: Node2Vec

3. **Similarity Calculation**
   - Cosine similarity matrix from node embeddings

4. **Model Training**
   - Models used: `GCN`, `GAT`, `GraphSAGE`
   - Dataset includes positive (real edges) and negative (random non-edges) samples

5. **Evaluation**
   - Metrics: `AUC-ROC`, `Average Precision`

6. **Visualization**
   - Interactive dashboard using `pyvis`
   - GUI input fields for model, threshold, graph, and K-value

---

# Technologies Used

- Python 3.x
- NetworkX
- PyTorch Geometric (`PyG`)
- Pyvis
- Scikit-learn
- NumPy, Pandas, Matplotlib

---

# File Structure

```
project-root/
│
├── data/                     # .net files (graph data)
├── embeddings/               # Node embeddings
├── models/                   # GCN, GAT, GraphSAGE implementations
├── utils/                    # Helper functions for preprocessing and evaluation
├── gui/                      # User interface components
├── main.py                   # Entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

# How to Run

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Application**
   ```bash
   python main.py
   ```

3. **Use the GUI**
   - Upload graph
   - Select model (GCN/GAT/GraphSAGE)
   - Set threshold and K-value
   - View predicted links and visual graph

---

# Results

- **GCN** performed best with highest AUC and Precision.
- Visualizations clearly showed enhanced graph structures with predicted links.
- Effective for applications in **recommendation systems**, **social media analysis**, and **cybersecurity**.

---

# Future Scope

- Real-time deployment for evolving networks
- Integration with Explainable AI (XAI)
- Extension to recommender systems using graph-based approaches
- Optimization using GPU acceleration (e.g., DGL, PyG)

---
