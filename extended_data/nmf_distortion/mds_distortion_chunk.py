#!/usr/bin/env python3

import os
import argparse
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr


def get_stress(d3, d2):
    denom = np.sum(d3 ** 2)
    if denom == 0:
        return np.nan
    return np.sqrt(np.sum((d3 - d2) ** 2) / denom)


def process_gene(gene, info_3d, output_dir):
    # 3D coords for this gene
    sub3d = info_3d.loc[
        info_3d["gene"] == gene,
        ["res", "x_coord", "y_coord", "z_coord"]
    ].copy()

    if sub3d.empty:
        return None

    # expected 2D file path
    two_d_file = os.path.join(output_dir, gene[0], f"{gene}_nmfinfo_final.csv")
    if not os.path.exists(two_d_file):
        return None

    df2d = pd.read_csv(two_d_file)

    # Handle possible unnamed residue column from saved index
    if "res" not in df2d.columns:
        first_col = df2d.columns[0]
        if str(first_col).startswith("Unnamed"):
            df2d = df2d.rename(columns={first_col: "res"})
        else:
            return None

    needed_cols = {"res", "x_axis", "y_axis"}
    if not needed_cols.issubset(df2d.columns):
        return None

    merged = (
        sub3d.merge(df2d[["res", "x_axis", "y_axis"]], on="res", how="inner")
        .dropna()
    )

    if merged.shape[0] < 3:
        return None

    coords_3d = merged[["x_coord", "y_coord", "z_coord"]].to_numpy()
    coords_2d = merged[["x_axis", "y_axis"]].to_numpy()

    d3 = pdist(coords_3d, metric="euclidean")
    d2 = pdist(coords_2d, metric="euclidean")

    if len(d3) == 0 or len(d2) == 0:
        return None

    # optional scale matching for 2D before stress
    mean_d2 = np.mean(d2)
    if mean_d2 == 0:
        d2_scaled = d2.copy()
    else:
        d2_scaled = d2 * (np.mean(d3) / mean_d2)

    pearson_r, pearson_p = pearsonr(d3, d2)
    spearman_r, spearman_p = spearmanr(d3, d2)
    stress = get_stress(d3, d2_scaled)

    return {
        "gene": gene,
        "n_residues": merged.shape[0],
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "stress": stress,
        "mean_3d_dist": np.mean(d3),
        "mean_2d_dist": np.mean(d2),
        "two_d_file": two_d_file,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coord3d", required=True, help="Path to 3Dcoord_allgenes.csv")
    parser.add_argument("--output-dir", required=True, help="Base dir containing A-Z subfolders of 2D files")
    parser.add_argument("--gene-list", required=True, help="Text file with one gene per line")
    parser.add_argument("--outdir", required=True, help="Directory for per-chunk outputs")
    parser.add_argument("--chunk-id", type=int, required=True, help="0-based chunk index")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Genes per chunk")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    with open(args.gene_list) as f:
        genes = [line.strip() for line in f if line.strip()]

    start = args.chunk_id * args.chunk_size
    end = min(start + args.chunk_size, len(genes))
    chunk_genes = genes[start:end]

    if len(chunk_genes) == 0:
        print(f"No genes in chunk {args.chunk_id}", flush=True)
        return

    outfile = os.path.join(args.outdir, f"chunk_{args.chunk_id:03d}_mds_distortion.csv")
    if os.path.exists(outfile):
        print(f"Chunk {args.chunk_id}: output exists, skipping", flush=True)
        return

    info_3d = pd.read_csv(args.coord3d)

    results = []
    n_missing_2d = 0
    n_no_3d = 0
    n_processed = 0

    for idx, gene in enumerate(chunk_genes, start=1):
        sub3d = info_3d.loc[info_3d["gene"] == gene]
        if sub3d.empty:
            n_no_3d += 1
            continue

        two_d_file = os.path.join(args.output_dir, gene[0], f"{gene}_nmfinfo_final.csv")
        if not os.path.exists(two_d_file):
            n_missing_2d += 1
            continue

        res = process_gene(gene, info_3d, args.output_dir)
        if res is not None:
            results.append(res)
            n_processed += 1

        if idx % 100 == 0 or idx == len(chunk_genes):
            print(
                f"Chunk {args.chunk_id}: {idx}/{len(chunk_genes)} checked | "
                f"processed={n_processed} missing_2d={n_missing_2d} no_3d={n_no_3d}",
                flush=True
            )

    out_df = pd.DataFrame(results)
    out_df.to_csv(outfile, index=False)
    print(f"Chunk {args.chunk_id}: wrote {outfile} with {len(out_df)} rows", flush=True)


if __name__ == "__main__":
    main()