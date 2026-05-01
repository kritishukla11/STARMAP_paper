import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv('../data/protein_trn_scores.csv')
required = {"protein", "pathway", "score"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"df is missing columns: {missing}")

df2 = df.dropna(subset=["protein", "pathway", "score"]).copy()

# Optional: if you have negative scores and want only positive associations:
# df2 = df2[df2["score"] > 0]

# Cast to categorical for fast integer encoding
prot = df2["protein"].astype("category")
path = df2["pathway"].astype("category")

pi = prot.cat.codes.to_numpy()
pj = path.cat.codes.to_numpy()
vals = df2["score"].to_numpy(dtype=np.float32)

n_prot = prot.cat.categories.size
n_path = path.cat.categories.size

print(f"Proteins: {n_prot:,}")
print(f"Pathways: {n_path:,}")
print(f"Rows: {len(df2):,}")

# Build sparse protein x pathway matrix X
X = sp.csr_matrix((vals, (pi, pj)), shape=(n_prot, n_path))

# Project to protein-protein adjacency (shared pathways only)
A = (X @ X.T).tocsr()
A.setdiag(0)
A.eliminate_zeros()

# -----------------------
# SPARSIFY: keep top-k neighbors per protein
# This is important for 15k nodes; otherwise the graph can be too dense.
# -----------------------
top_k = 25  # tune: 25–100 are typical
A = A.tolil()  # easier row-wise editing

for i in range(n_prot):
    row = A.data[i]
    cols = A.rows[i]
    if len(row) <= top_k:
        continue
    # keep indices of top_k largest weights in this row
    keep_idx = np.argpartition(row, -top_k)[-top_k:]
    A.data[i] = [row[j] for j in keep_idx]
    A.rows[i] = [cols[j] for j in keep_idx]

A = A.tocsr()
A.eliminate_zeros()

# Make symmetric (top-k is directed-like unless symmetrized)
A = A.maximum(A.T)
A.eliminate_zeros()

# Build NetworkX graph (proteins only)
proteins = prot.cat.categories.to_list()
G = nx.from_scipy_sparse_array(A)
G = nx.relabel_nodes(G, dict(enumerate(proteins)))

print(f"Graph nodes: {G.number_of_nodes():,}")
print(f"Graph edges: {G.number_of_edges():,}")

# -----------------------
# METRICS
# -----------------------
metrics = pd.DataFrame(index=G.nodes())
metrics["degree"] = pd.Series(dict(G.degree()), dtype=float)
metrics["weighted_degree"] = pd.Series(dict(G.degree(weight="weight")), dtype=float)

# Eigenvector can fail to converge on some graphs; handle gracefully
try:
    ev = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
    metrics["eigenvector"] = pd.Series(ev, dtype=float)
except Exception as e:
    print(f"Eigenvector centrality failed: {e}")
    metrics["eigenvector"] = np.nan

# Betweenness is expensive; compute on largest connected component only
# and (optionally) on a downsampled edge set.
largest_cc = max(nx.connected_components(G), key=len)
H = G.subgraph(largest_cc).copy()

edges = nx.to_pandas_edgelist(H)
edges.to_csv("../output/protein_network_edges.csv", index=False)

print(f"Largest connected component nodes: {H.number_of_nodes():,}")
print("Computing betweenness on largest component (can be slow)...")

try:
    bt = nx.betweenness_centrality(H, weight="weight")
    metrics.loc[H.nodes(), "betweenness"] = pd.Series(bt, dtype=float)
except Exception as e:
    print(f"Betweenness failed: {e}")
    metrics["betweenness"] = np.nan

# Top 10 tables
top10_weighted = metrics.sort_values("weighted_degree", ascending=False).head(10)
top10_degree = metrics.sort_values("degree", ascending=False).head(10)
top10_betw = metrics.sort_values("betweenness", ascending=False).head(10)

print("\nTop 10 by weighted degree:")
print(top10_weighted)

print("\nTop 10 by degree:")
print(top10_degree)

print("\nTop 10 by betweenness:")
print(top10_betw)

# Save metrics to CSV
metrics.sort_values("weighted_degree", ascending=False).to_csv("../output/protein_network_metrics.csv")

# -----------------------
# PNG VISUALIZATION
# Draw largest CC only for readability
# Node size ~ weighted degree
# -----------------------
plt.figure(figsize=(12, 12))

# Layout: spring works well; for big graphs, iterations modest
pos = nx.spring_layout(H, seed=42, weight="weight", k=None, iterations=100)

# Node sizes
wdeg = dict(H.degree(weight="weight"))
wdeg_vals = np.array(list(wdeg.values()), dtype=float)
if len(wdeg_vals) > 0:
    # normalize for display
    wdeg_norm = (wdeg_vals - wdeg_vals.min()) / (wdeg_vals.max() - wdeg_vals.min() + 1e-9)
else:
    wdeg_norm = wdeg_vals

node_sizes = {
    n: 50 + 1000 * wdeg_norm[i]
    for i, n in enumerate(wdeg.keys())
}

# Edge widths (cap for readability)
edge_weights = np.array([d["weight"] for _, _, d in H.edges(data=True)], dtype=float)
if len(edge_weights) > 0:
    ew_norm = (edge_weights - edge_weights.min()) / (edge_weights.max() - edge_weights.min() + 1e-9)
else:
    ew_norm = edge_weights
edge_widths = 0.2 + 2.5 * ew_norm

nx.draw_networkx_nodes(
    H, pos,
    node_size=[node_sizes[n] for n in H.nodes()],
    alpha=0.85
)
nx.draw_networkx_edges(
    H, pos,
    width=edge_widths,
    alpha=0.25
)

plt.axis("off")
plt.tight_layout()
plt.savefig("../output/protein_shared_trn_network.png", dpi=300)
plt.close()
