import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

df = pd.read_csv("../data/protein_trn_scores.csv")
required = {"protein", "pathway", "score"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"df is missing columns: {missing}")

df2 = df.dropna(subset=["protein", "pathway", "score"]).copy()

# Optional: only positive associations
# df2 = df2[df2["score"] > 0]

# Cast to categorical for fast integer encoding
prot = df2["protein"].astype("category")
path = df2["pathway"].astype("category")

pi = prot.cat.codes.to_numpy()
pj = path.cat.codes.to_numpy()

# -----------------------
# CHOOSE EDGE DEFINITION
# -----------------------
# 1) "weighted-by-score" projection:
vals = df2["score"].to_numpy(dtype=np.float32)

# 2) "shared-protein count" projection (uncomment to use):
# vals = np.ones(len(df2), dtype=np.float32)

n_prot = prot.cat.categories.size
n_path = path.cat.categories.size

print(f"Proteins: {n_prot:,}")
print(f"Pathways: {n_path:,}")
print(f"Rows: {len(df2):,}")

# Build sparse protein x pathway matrix X
X = sp.csr_matrix((vals, (pi, pj)), shape=(n_prot, n_path))

# -----------------------
# PROJECT TO PATHWAY-PATHWAY ADJACENCY
# -----------------------
B = (X.T @ X).tocsr()
B.setdiag(0)
B.eliminate_zeros()

# -----------------------
# SPARSIFY: keep top-k neighbors per pathway
# -----------------------
top_k = 25  # tune: 25–100 typical
B = B.tolil()

for i in range(n_path):
    row = B.data[i]
    cols = B.rows[i]
    if len(row) <= top_k:
        continue
    keep_idx = np.argpartition(row, -top_k)[-top_k:]
    B.data[i] = [row[j] for j in keep_idx]
    B.rows[i] = [cols[j] for j in keep_idx]

B = B.tocsr()
B.eliminate_zeros()

# Make symmetric (top-k is directed-like unless symmetrized)
B = B.maximum(B.T)
B.eliminate_zeros()

# Build NetworkX graph (pathways only)
pathways = path.cat.categories.to_list()
G = nx.from_scipy_sparse_array(B)
G = nx.relabel_nodes(G, dict(enumerate(pathways)))

print(f"Graph nodes: {G.number_of_nodes():,}")
print(f"Graph edges: {G.number_of_edges():,}")

# -----------------------
# METRICS
# -----------------------
metrics = pd.DataFrame(index=G.nodes())
metrics["degree"] = pd.Series(dict(G.degree()), dtype=float)
metrics["weighted_degree"] = pd.Series(dict(G.degree(weight="weight")), dtype=float)

try:
    ev = nx.eigenvector_centrality(G, weight="weight", max_iter=2000)
    metrics["eigenvector"] = pd.Series(ev, dtype=float)
except Exception as e:
    print(f"Eigenvector centrality failed: {e}")
    metrics["eigenvector"] = np.nan

largest_cc = max(nx.connected_components(G), key=len)
H = G.subgraph(largest_cc).copy()

edges = nx.to_pandas_edgelist(H)
edges.to_csv("../output/trn_network_edges.csv", index=False)

print(f"Largest connected component nodes: {H.number_of_nodes():,}")
print("Computing betweenness on largest component (can be slow)...")

try:
    bt = nx.betweenness_centrality(H, weight="weight")
    metrics.loc[H.nodes(), "betweenness"] = pd.Series(bt, dtype=float)
except Exception as e:
    print(f"Betweenness failed: {e}")
    metrics["betweenness"] = np.nan

top10_weighted = metrics.sort_values("weighted_degree", ascending=False).head(10)
top10_degree = metrics.sort_values("degree", ascending=False).head(10)
top10_betw = metrics.sort_values("betweenness", ascending=False).head(10)

print("\nTop 10 by weighted degree:")
print(top10_weighted)

print("\nTop 10 by degree:")
print(top10_degree)

print("\nTop 10 by betweenness:")
print(top10_betw)

metrics.sort_values("weighted_degree", ascending=False).to_csv("../output/trn_network_metrics.csv")

# -----------------------
# PNG VISUALIZATION (largest CC)
# -----------------------
plt.figure(figsize=(12, 12))
pos = nx.spring_layout(H, seed=42, weight="weight", iterations=100)

wdeg = dict(H.degree(weight="weight"))
wdeg_vals = np.array(list(wdeg.values()), dtype=float)
if len(wdeg_vals) > 0:
    wdeg_norm = (wdeg_vals - wdeg_vals.min()) / (wdeg_vals.max() - wdeg_vals.min() + 1e-9)
else:
    wdeg_norm = wdeg_vals

node_sizes = {n: 50 + 1000 * wdeg_norm[i] for i, n in enumerate(wdeg.keys())}

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
plt.savefig("../output/trn_shared_protein_network.png", dpi=300)
plt.close()
