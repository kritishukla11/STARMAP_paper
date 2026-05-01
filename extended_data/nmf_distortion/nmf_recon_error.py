#!/usr/bin/env python3

import os
import argparse
import pandas as pd
import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error


def min_max_scaling(column):
    col_min = column.min()
    col_max = column.max()
    denom = col_max - col_min
    if denom == 0:
        return pd.Series(np.zeros(len(column)), index=column.index)
    return (column - col_min) / denom


def rank_mean_normalize(df):
    df_num = df.apply(pd.to_numeric, errors="coerce")

    ranked_first = df_num.rank(method="first")
    stacked_first = ranked_first.stack()

    rank_mean = df_num.stack().groupby(stacked_first.astype(int)).mean()

    ranked_min = df_num.rank(method="min").stack().astype(int)
    out = ranked_min.map(rank_mean).unstack()

    return out


def process_gene(gene, info, k_values, max_iter):
    df = info.loc[info["gene"] == gene, ["x_coord", "y_coord", "z_coord", "res"]].copy()

    if df.empty:
        return [{
            "gene": gene,
            "k": np.nan,
            "mse": np.nan,
            "n_residues": 0,
            "n_features": 0,
            "status": "no_rows"
        }]

    ge = df.set_index("res")[["x_coord", "y_coord", "z_coord"]]
    ge = ge.loc[:, ge.nunique(dropna=True) > 1]

    if ge.shape[1] == 0:
        return [{
            "gene": gene,
            "k": np.nan,
            "mse": np.nan,
            "n_residues": ge.shape[0],
            "n_features": 0,
            "status": "no_variable_features"
        }]

    v = rank_mean_normalize(ge)

    for col in v.columns:
        v[col] = min_max_scaling(v[col])

    v = v.dropna(axis=0, how="any").dropna(axis=1, how="any")

    n_residues = v.shape[0]
    n_features = v.shape[1]

    if n_residues == 0 or n_features == 0:
        return [{
            "gene": gene,
            "k": np.nan,
            "mse": np.nan,
            "n_residues": n_residues,
            "n_features": n_features,
            "status": "empty_after_processing"
        }]

    results = []

    for k in k_values:
        if k > min(n_residues, n_features):
            results.append({
                "gene": gene,
                "k": k,
                "mse": np.nan,
                "n_residues": n_residues,
                "n_features": n_features,
                "status": "k_too_large"
            })
            continue

        try:
            model = NMF(
                n_components=k,
                init="random",
                random_state=0,
                max_iter=max_iter
            )
            w = model.fit_transform(v)
            h = model.components_
            recon = w @ h
            mse = mean_squared_error(v.values, recon)

            results.append({
                "gene": gene,
                "k": k,
                "mse": mse,
                "n_residues": n_residues,
                "n_features": n_features,
                "status": "ok"
            })

        except Exception as e:
            results.append({
                "gene": gene,
                "k": k,
                "mse": np.nan,
                "n_residues": n_residues,
                "n_features": n_features,
                "status": f"error: {type(e).__name__}"
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gene-list", required=True)
    parser.add_argument("--info-csv", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--chunk-id", type=int, required=True)
    parser.add_argument("--chunk-size", type=int, default=1000)
    parser.add_argument("--k-min", type=int, default=3)
    parser.add_argument("--k-max", type=int, default=6)
    parser.add_argument("--max-iter", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.gene_list) as f:
        genes = [line.strip() for line in f if line.strip()]

    start = args.chunk_id * args.chunk_size
    end = min(start + args.chunk_size, len(genes))
    chunk_genes = genes[start:end]

    if len(chunk_genes) == 0:
        print(f"No genes in chunk {args.chunk_id}")
        return

    outfile = os.path.join(args.outdir, f"chunk_{args.chunk_id:03d}_recon_error.csv")
    if os.path.exists(outfile):
        print(f"Chunk {args.chunk_id}: output exists, skipping")
        return

    info = pd.read_csv(args.info_csv)
    k_values = list(range(args.k_min, args.k_max + 1))

    all_results = []

    for idx, gene in enumerate(chunk_genes, start=1):
        all_results.extend(process_gene(gene, info, k_values, args.max_iter))

        if idx % 100 == 0 or idx == len(chunk_genes):
            print(
                f"Chunk {args.chunk_id}: processed {idx}/{len(chunk_genes)} genes",
                flush=True
            )

    pd.DataFrame(all_results).to_csv(outfile, index=False)
    print(f"Chunk {args.chunk_id}: wrote {outfile}", flush=True)


if __name__ == "__main__":
    main()