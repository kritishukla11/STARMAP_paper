#!/usr/bin/env python3

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def get_best_r2_from_file(csv_file):
    try:
        sub = pd.read_csv(csv_file)

        # Use index as adjusted_rank if adjusted_rank is not present
        if "adjusted_rank" not in sub.columns:
            if "index" in sub.columns:
                sub["adjusted_rank"] = sub["index"]
            else:
                return None

        if "confidence" not in sub.columns:
            return None

        # Prep data
        sub["adjusted_rank"] = pd.to_numeric(sub["adjusted_rank"], errors="coerce")
        sub["confidence"] = pd.to_numeric(sub["confidence"], errors="coerce")
        sub = sub.dropna(subset=["adjusted_rank", "confidence"]).reset_index(drop=True)

        # Need enough points
        if len(sub) < 3:
            return None

        x = sub["adjusted_rank"].to_numpy(dtype=float) + 1
        y = sub["confidence"].to_numpy(dtype=float)

        # Sort and anchor
        order = np.argsort(x)
        x = x[order]
        y = y[order]

        # Skip constant y
        if np.allclose(y, y[0]):
            return None

        x0, y0 = x[0], y[0]

        # Anchored candidate models
        def exp_decay(x, b, c):
            a = (y0 - c) / np.exp(-b * x0)
            return a * np.exp(-b * x) + c

        def logistic(x, k, xmid, c):
            L = (y0 - c) * (1 + np.exp(k * (x0 - xmid)))
            return c + L / (1 + np.exp(k * (x - xmid)))

        def linear(x, m):
            return m * x + (y0 - m * x0)

        candidates = []

        # Exponential
        try:
            popt, _ = curve_fit(
                exp_decay,
                x,
                y,
                p0=[0.01, np.min(y)],
                bounds=([0, -np.inf], [np.inf, np.inf]),
                maxfev=20000
            )
            yhat = exp_decay(x, *popt)
            r2 = r2_score(y, yhat)
            if np.isfinite(r2):
                candidates.append(("Exponential", r2))
        except Exception:
            pass

        # Logistic
        try:
            popt, _ = curve_fit(
                logistic,
                x,
                y,
                p0=[0.01, np.median(x), np.min(y)],
                maxfev=20000
            )
            yhat = logistic(x, *popt)
            r2 = r2_score(y, yhat)
            if np.isfinite(r2):
                candidates.append(("Logistic", r2))
        except Exception:
            pass

        # Linear
        try:
            popt, _ = curve_fit(
                linear,
                x,
                y,
                p0=[-0.001],
                maxfev=10000
            )
            yhat = linear(x, *popt)
            r2 = r2_score(y, yhat)
            if np.isfinite(r2):
                candidates.append(("Linear", r2))
        except Exception:
            pass

        if not candidates:
            return None

        best_model, best_r2 = max(candidates, key=lambda z: z[1])

        # Protein name
        if "protein" in sub.columns and sub["protein"].notna().any():
            protein = str(sub["protein"].dropna().iloc[0])
        else:
            protein = os.path.basename(csv_file).replace("_tahoe_confidence_metrics.csv", "")

        return {
            "protein": protein,
            "r2": best_r2,
            "best_model": best_model
        }

    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Compute best-fit R^2 for *_tahoe_confidence_metrics.csv files"
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing *_tahoe_confidence_metrics.csv files"
    )
    parser.add_argument(
        "--output_csv",
        default="protein_mlp_r2.csv",
        help="Output CSV filename"
    )
    args = parser.parse_args()

    pattern = os.path.join(args.input_dir, "*_tahoe_confidence_metrics.csv")
    all_files = sorted(glob.glob(pattern))

    if not all_files:
        print(f"No matching files found in {args.input_dir}")
        return

    results = []
    for i, f in enumerate(all_files, 1):
        if i % 100 == 0:
            print(f"Processed {i}/{len(all_files)} files")
        res = get_best_r2_from_file(f)
        if res is not None:
            results.append(res)

    if not results:
        print("No valid results found.")
        return

    results_df = pd.DataFrame(results)
    results_df[["protein", "r2"]].to_csv(args.output_csv, index=False)

    print(f"Processed {len(all_files)} files total")
    print(f"Saved {len(results_df)} protein results to {args.output_csv}")


if __name__ == "__main__":
    main()