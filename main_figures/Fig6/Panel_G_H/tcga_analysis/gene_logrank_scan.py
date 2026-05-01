import os
import argparse
import numpy as np
import pandas as pd
from lifelines.statistics import logrank_test


def evaluate_gene_day_cutoffs(
    gene,
    struct_df,
    surv_df,
    min_day=365,
    min_group_size=5,
    step=90,
    per_gene_day_outdir=None,
    overwrite_gene_files=False
):
    sub_mut = struct_df[struct_df["gene"] == gene]
    sub_other = struct_df[struct_df["gene"] != gene]

    list_mut = sorted(set(sub_mut["Tumor"].astype(str).str[:12]))
    list_other_overlap = sorted(set(sub_other["Tumor"].astype(str).str[:12]))
    list_other = [x for x in list_other_overlap if x not in set(list_mut)]

    empty_summary = {
        "gene": gene,
        "day_for_min_pval": np.nan,
        "min_pval": np.nan,
        "all_sig_days": "",
        "n_mut": len(list_mut),
        "n_other": len(list_other)
    }

    if len(list_mut) < min_group_size or len(list_other) < min_group_size:
        return empty_summary

    patient_to_group = {pid: "mut" for pid in list_mut}
    patient_to_group.update({pid: "other" for pid in list_other})

    tmp = surv_df.copy()
    tmp["group"] = tmp["_PATIENT"].map(patient_to_group)
    tmp = tmp.dropna(subset=["group"]).copy()

    gcounts = tmp["group"].value_counts()
    if ("mut" not in gcounts) or ("other" not in gcounts):
        return empty_summary

    max_day = int(tmp["OS.time"].max())
    if max_day < min_day:
        return empty_summary

    gene_day_file = None
    if per_gene_day_outdir is not None:
        os.makedirs(per_gene_day_outdir, exist_ok=True)
        gene_day_file = os.path.join(per_gene_day_outdir, f"{gene}.csv")

    # if per-gene file already exists and we do not want to overwrite,
    # load it and just compute the summary from that
    if (
        gene_day_file is not None
        and os.path.exists(gene_day_file)
        and not overwrite_gene_files
    ):
        try:
            day_df = pd.read_csv(gene_day_file)
            if not day_df.empty and {"day", "pval"}.issubset(day_df.columns):
                day_df = day_df.dropna(subset=["day", "pval"]).copy()
                if not day_df.empty:
                    min_idx = day_df["pval"].idxmin()
                    min_day_cutoff = int(day_df.loc[min_idx, "day"])
                    min_pval = float(day_df.loc[min_idx, "pval"])
                    sig_days = day_df.loc[day_df["pval"] < 0.05, "day"].astype(int).tolist()

                    return {
                        "gene": gene,
                        "day_for_min_pval": min_day_cutoff,
                        "min_pval": min_pval,
                        "all_sig_days": ";".join(map(str, sig_days)) if len(sig_days) > 0 else "",
                        "n_mut": len(list_mut),
                        "n_other": len(list_other)
                    }
        except Exception:
            pass

    day_results = []

    for day_cutoff in range(min_day, max_day + 1, step):
        tdf = tmp.copy()
        tdf["OS.time.cut"] = np.minimum(tdf["OS.time"], day_cutoff)
        tdf["OS.cut"] = np.where(tdf["OS.time"] > day_cutoff, 0, tdf["OS"])

        g1 = tdf[tdf["group"] == "mut"]
        g2 = tdf[tdf["group"] == "other"]

        if len(g1) < min_group_size or len(g2) < min_group_size:
            continue

        try:
            result = logrank_test(
                g1["OS.time.cut"],
                g2["OS.time.cut"],
                event_observed_A=g1["OS.cut"],
                event_observed_B=g2["OS.cut"]
            )
            pval = result.p_value
            test_stat = result.test_statistic
        except Exception:
            pval = np.nan
            test_stat = np.nan

        day_results.append({
            "gene": gene,
            "day": int(day_cutoff),
            "pval": pval,
            "test_statistic": test_stat,
            "n_mut": len(g1),
            "n_other": len(g2),
            "mut_events": int(g1["OS.cut"].sum()),
            "other_events": int(g2["OS.cut"].sum())
        })

    if len(day_results) == 0:
        return empty_summary

    day_df = pd.DataFrame(day_results).dropna(subset=["pval"]).copy()

    if day_df.empty:
        return empty_summary

    # save full per-gene per-day table
    if gene_day_file is not None:
        day_df.to_csv(gene_day_file, index=False)

    min_idx = day_df["pval"].idxmin()
    min_day_cutoff = int(day_df.loc[min_idx, "day"])
    min_pval = float(day_df.loc[min_idx, "pval"])
    sig_days = day_df.loc[day_df["pval"] < 0.05, "day"].astype(int).tolist()

    return {
        "gene": gene,
        "day_for_min_pval": min_day_cutoff,
        "min_pval": min_pval,
        "all_sig_days": ";".join(map(str, sig_days)) if len(sig_days) > 0 else "",
        "n_mut": len(list_mut),
        "n_other": len(list_other)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--structure_df", required=True)
    parser.add_argument("--survival_df", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--task_id", type=int, required=True)
    parser.add_argument("--n_tasks", type=int, required=True)
    parser.add_argument("--min_day", type=int, default=365)
    parser.add_argument("--min_group_size", type=int, default=5)
    parser.add_argument("--step", type=int, default=90)
    parser.add_argument("--overwrite_gene_files", action="store_true")
    args = parser.parse_args()

    summary_dir = os.path.join(args.outdir, "summary")
    per_gene_day_dir = os.path.join(args.outdir, "per_gene_days")

    os.makedirs(summary_dir, exist_ok=True)
    os.makedirs(per_gene_day_dir, exist_ok=True)

    df = pd.read_csv(args.structure_df)
    surv = pd.read_csv(args.survival_df)

    df["Tumor"] = df["Tumor"].astype(str).str.replace("_", "-", regex=False)
    df["gene"] = df["gene"].astype(str)

    surv["_PATIENT"] = surv["_PATIENT"].astype(str).str[:12]
    surv = surv[["_PATIENT", "OS.time", "OS"]].copy()
    surv = surv.dropna(subset=["_PATIENT", "OS.time", "OS"])
    surv = surv.drop_duplicates(subset=["_PATIENT"]).copy()
    surv["OS.time"] = pd.to_numeric(surv["OS.time"], errors="coerce")
    surv["OS"] = pd.to_numeric(surv["OS"], errors="coerce")
    surv = surv.dropna(subset=["OS.time", "OS"]).copy()
    surv["OS"] = surv["OS"].astype(int)

    genes = sorted(df["gene"].dropna().unique())
    gene_chunks = np.array_split(genes, args.n_tasks)
    my_genes = list(gene_chunks[args.task_id])

    print(f"Task {args.task_id} processing {len(my_genes)} genes")

    results = []
    for i, gene in enumerate(my_genes, start=1):
        if i % 25 == 0 or i == len(my_genes):
            print(f"Task {args.task_id}: {i}/{len(my_genes)} genes done")

        out = evaluate_gene_day_cutoffs(
            gene=gene,
            struct_df=df,
            surv_df=surv,
            min_day=args.min_day,
            min_group_size=args.min_group_size,
            step=args.step,
            per_gene_day_outdir=per_gene_day_dir,
            overwrite_gene_files=args.overwrite_gene_files
        )
        results.append(out)

    out_df = pd.DataFrame(results)
    out_df = out_df.sort_values("min_pval", ascending=True, na_position="last")

    outfile = os.path.join(summary_dir, f"gene_logrank_chunk_{args.task_id}.csv")
    out_df.to_csv(outfile, index=False)
    print(f"Saved summary: {outfile}")


if __name__ == "__main__":
    main()