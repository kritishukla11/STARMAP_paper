import argparse
import numpy as np
import pandas as pd
import scipy.sparse as sp
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities, modularity


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True, help="Input CSV with protein/drug/score columns")
    parser.add_argument("--output_prefix", required=True, help="Prefix for output files")
    parser.add_argument("--top_k", type=int, default=25, help="Top-k neighbors per protein after projection")
    parser.add_argument("--n_null", type=int, default=250, help="Number of null iterations per null type")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def gini(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan
    if np.allclose(x, 0):
        return 0.0
    x = np.sort(x)
    n = len(x)
    return (2 * np.sum((np.arange(1, n + 1) * x)) / (n * np.sum(x))) - (n + 1) / n


def build_projected_graph(df_in, top_k=25):
    prot = df_in["protein"].astype("category")
    drug = df_in["drug"].astype("category")

    pi = prot.cat.codes.to_numpy()
    dj = drug.cat.codes.to_numpy()
    vals = df_in["score"].to_numpy(dtype=np.float32)

    n_prot = prot.cat.categories.size
    n_drug = drug.cat.categories.size

    X = sp.csr_matrix((vals, (pi, dj)), shape=(n_prot, n_drug))

    A = (X @ X.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()

    A = A.tolil()
    for i in range(n_prot):
        row = A.data[i]
        cols = A.rows[i]
        if len(row) <= top_k:
            continue
        keep_idx = np.argpartition(row, -top_k)[-top_k:]
        A.data[i] = [row[j] for j in keep_idx]
        A.rows[i] = [cols[j] for j in keep_idx]

    A = A.tocsr()
    A.eliminate_zeros()

    A = A.maximum(A.T)
    A.eliminate_zeros()

    proteins = prot.cat.categories.to_list()
    G = nx.from_scipy_sparse_array(A)
    G = nx.relabel_nodes(G, dict(enumerate(proteins)))
    return G


def compute_graph_stats(G):
    stats = {}

    stats["n_nodes"] = G.number_of_nodes()
    stats["n_edges"] = G.number_of_edges()
    stats["density"] = nx.density(G)

    if G.number_of_nodes() > 0 and G.number_of_edges() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc).copy()
        stats["lcc_frac"] = H.number_of_nodes() / G.number_of_nodes()
    else:
        H = G.copy()
        stats["lcc_frac"] = np.nan

    wdeg = np.array([d for _, d in G.degree(weight="weight")], dtype=float)
    stats["weighted_degree_mean"] = np.mean(wdeg) if len(wdeg) else np.nan
    stats["weighted_degree_gini"] = gini(wdeg)
    stats["weighted_degree_cv"] = (np.std(wdeg) / np.mean(wdeg)) if np.mean(wdeg) > 0 else np.nan

    try:
        stats["weighted_clustering"] = nx.average_clustering(G, weight="weight")
    except Exception:
        stats["weighted_clustering"] = np.nan

    try:
        if H.number_of_nodes() > 2 and H.number_of_edges() > 0:
            comms = list(greedy_modularity_communities(H, weight="weight"))
            stats["modularity"] = modularity(H, comms, weight="weight")
            stats["n_communities"] = len(comms)
        else:
            stats["modularity"] = np.nan
            stats["n_communities"] = np.nan
    except Exception:
        stats["modularity"] = np.nan
        stats["n_communities"] = np.nan

    return stats


def null_A_global_drug_permute(df_in, rng):
    out = df_in.copy()
    out["drug"] = rng.permutation(out["drug"].to_numpy())
    return out


def null_C_within_protein_score_shuffle(df_in, rng):
    out = df_in.copy()
    shuffled_scores = np.empty(len(out), dtype=float)

    for protein, idx in out.groupby("protein").groups.items():
        idx = np.array(list(idx))
        vals = out.loc[idx, "score"].to_numpy()
        shuffled_scores[idx] = rng.permutation(vals)

    out["score"] = shuffled_scores
    return out


def null_B_degree_preserving_rewire(G_real, rng, nswap_factor=10):
    G = nx.Graph()
    G.add_nodes_from(G_real.nodes())
    G.add_edges_from(G_real.edges())

    nswap = max(10, nswap_factor * G.number_of_edges())
    max_tries = max(100, 20 * nswap)

    try:
        nx.double_edge_swap(G, nswap=nswap, max_tries=max_tries, seed=int(rng.integers(1_000_000_000)))
    except Exception:
        pass

    original_weights = np.array([d["weight"] for _, _, d in G_real.edges(data=True)], dtype=float)
    if len(original_weights) != G.number_of_edges():
        original_weights = rng.choice(original_weights, size=G.number_of_edges(), replace=True)

    perm_weights = rng.permutation(original_weights)
    for (u, v), w in zip(G.edges(), perm_weights):
        G[u][v]["weight"] = float(w)

    return G


def empirical_summary(obs, x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0 or not np.isfinite(obs):
        return np.nan, np.nan, np.nan, np.nan, np.nan
    z = (obs - np.mean(x)) / (np.std(x, ddof=1) + 1e-12)
    p_high = (np.sum(x >= obs) + 1) / (len(x) + 1)
    p_low = (np.sum(x <= obs) + 1) / (len(x) + 1)
    return np.mean(x), np.std(x, ddof=1), z, p_high, p_low


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    df = pd.read_csv(args.input_csv)
    required = {"protein", "drug", "score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    df = df.dropna(subset=["protein", "drug", "score"]).copy()
    df["score"] = df["score"].astype(float)

    print(f"Rows: {len(df):,}")
    print(f"Proteins: {df['protein'].nunique():,}")
    print(f"Drugs: {df['drug'].nunique():,}")

    G_obs = build_projected_graph(df, top_k=args.top_k)
    obs_stats = compute_graph_stats(G_obs)
    pd.DataFrame([obs_stats]).to_csv(f"{args.output_prefix}_observed_graph_stats.csv", index=False)

    results = []

    for i in range(args.n_null):
        print(f"Iteration {i+1}/{args.n_null}")

        df_A = null_A_global_drug_permute(df, rng)
        G_A = build_projected_graph(df_A, top_k=args.top_k)
        stats_A = compute_graph_stats(G_A)
        stats_A["null_type"] = "A_global_drug_permute"
        stats_A["iter"] = i
        results.append(stats_A)

        df_C = null_C_within_protein_score_shuffle(df, rng)
        G_C = build_projected_graph(df_C, top_k=args.top_k)
        stats_C = compute_graph_stats(G_C)
        stats_C["null_type"] = "C_within_protein_score_shuffle"
        stats_C["iter"] = i
        results.append(stats_C)

        G_B = null_B_degree_preserving_rewire(G_obs, rng)
        stats_B = compute_graph_stats(G_B)
        stats_B["null_type"] = "B_degree_preserving_rewire"
        stats_B["iter"] = i
        results.append(stats_B)

    null_df = pd.DataFrame(results)
    null_df.to_csv(f"{args.output_prefix}_null_graph_stats.csv", index=False)

    metrics_to_compare = [
        "weighted_clustering",
        "modularity",
        "lcc_frac",
        "weighted_degree_gini",
        "weighted_degree_cv",
    ]

    summary_rows = []
    for null_type, sub in null_df.groupby("null_type"):
        for metric in metrics_to_compare:
            obs = obs_stats.get(metric, np.nan)
            x = sub[metric].dropna().to_numpy(dtype=float)
            null_mean, null_sd, z, p_high, p_low = empirical_summary(obs, x)

            summary_rows.append({
                "null_type": null_type,
                "metric": metric,
                "observed": obs,
                "null_mean": null_mean,
                "null_sd": null_sd,
                "z_score": z,
                "empirical_p_high": p_high,
                "empirical_p_low": p_low
            })

    pd.DataFrame(summary_rows).to_csv(f"{args.output_prefix}_null_comparison_summary.csv", index=False)

    print("Done.")


if __name__ == "__main__":
    main()