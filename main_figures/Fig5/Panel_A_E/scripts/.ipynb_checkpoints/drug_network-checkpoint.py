import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

# -----------------------
# INPUT: protein, drug, score
# -----------------------
df = pd.read_csv("../data/drug_protein_scores.csv")
required = {"protein", "drug", "score"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"df is missing columns: {missing}")

df2 = df.dropna(subset=["protein", "drug", "score"]).copy()

# Optional: only positive associations
# df2 = df2[df2["score"] > 0]

# Optional: aggregate duplicates (same protein-drug pair) before projection
# (useful if you have repeated measurements)
df2 = (
    df2.groupby(["protein", "drug"], as_index=False)["score"]
       .sum()   # or .mean()
)

# Cast to categorical for fast integer encoding
prot = df2["protein"].astype("category")
drug = df2["drug"].astype("category")

pi = prot.cat.codes.to_numpy()
dj = drug.cat.codes.to_numpy()

# -----------------------
# CHOOSE EDGE DEFINITION (drug-drug projection)
# -----------------------
# 1) "weighted-by-score" projection: (shared proteins with product of scores)
vals = df2["score"].to_numpy(dtype=np.float32)

# 2) "shared-protein count" projection (uncomment to use):
# vals = np.ones(len(df2), dtype=np.float32)

n_prot = prot.cat.categories.size
n_drug = drug.cat.categories.size

print(f"Proteins: {n_prot:,}")
print(f"Drugs: {n_drug:,}")
print(f"Rows: {len(df2):,}")

# Build sparse protein x drug matrix X
# rows = proteins, cols = drugs
X = sp.csr_matrix((vals, (pi, dj)), shape=(n_prot, n_drug))

# -----------------------
# PROJECT TO DRUG-DRUG ADJACENCY
# -----------------------
# Drug-drug similarity via shared proteins:
# B[j,k] = sum_over_proteins X[p,j] * X[p,k]
B = (X.T @ X).tocsr()
B.setdiag(0)
B.eliminate_zeros()

# -----------------------
# SPARSIFY: keep top-k neighbors per drug
# -----------------------
top_k = 25  # tune: 25–100 typical
B = B.tolil()

for i in range(n_drug):
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

# Build NetworkX graph (drugs only)
drugs = drug.cat.categories.to_list()
G = nx.from_scipy_sparse_array(B)
G = nx.relabel_nodes(G, dict(enumerate(drugs)))

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

if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()

    edges = nx.to_pandas_edgelist(H)
    edges.to_csv("../output/drug_network_edges.csv", index=False)

    print(f"Largest connected component nodes: {H.number_of_nodes():,}")
    print("Computing betweenness on largest component (can be slow)...")

    try:
        bt = nx.betweenness_centrality(H, weight="weight")
        metrics.loc[H.nodes(), "betweenness"] = pd.Series(bt, dtype=float)
    except Exception as e:
        print(f"Betweenness failed: {e}")
        metrics["betweenness"] = np.nan
else:
    H = G.copy()
    metrics["betweenness"] = np.nan

top10_weighted = metrics.sort_values("weighted_degree", ascending=False).head(10)
top10_degree = metrics.sort_values("degree", ascending=False).head(10)
top10_betw = metrics.sort_values("betweenness", ascending=False).head(10)

print("\nTop 10 by weighted degree (drugs):")
print(top10_weighted)

print("\nTop 10 by degree (drugs):")
print(top10_degree)

print("\nTop 10 by betweenness (drugs):")
print(top10_betw)

metrics.sort_values("weighted_degree", ascending=False).to_csv("../output/drug_network_metrics.csv")

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
plt.savefig("../output/drug_shared_protein_network.png", dpi=300)
plt.close()