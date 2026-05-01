#!/usr/bin/env python3
import os
import argparse
import pandas as pd
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef

def oversample_minority(df, class_col="class", random_state=42):
    class_counts = df[class_col].value_counts()
    if len(class_counts) < 2:
        return df

    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    df_majority = df[df[class_col] == majority_class]
    df_minority = df[df[class_col] == minority_class]

    if len(df_minority) == 0 or len(df_majority) == 0:
        return df

    df_minority_upsampled = resample(
        df_minority, replace=True, n_samples=len(df_majority), random_state=random_state
    )

    df_balanced = (
        pd.concat([df_majority, df_minority_upsampled])
          .sample(frac=1, random_state=random_state)
          .reset_index(drop=True)
    )
    return df_balanced

def run_one_trn(trn_name, in_dir, out_dir, random_state=42, cv=3, n_estimators=50):
    in_path = os.path.join(in_dir, f"{trn_name}_ML_outputs.csv")
    out_path = os.path.join(out_dir, f"{trn_name}_ML_outputs.csv")

    os.makedirs(out_dir, exist_ok=True)

    if os.path.exists(out_path):
        return "skipped_exists"

    if not os.path.exists(in_path):
        return "missing_input"

    df = pd.read_csv(in_path)

    needed = ["x", "y", "z", "cluster", "dist", "class"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        return f"missing_cols:{','.join(missing)}"

    df_balanced = oversample_minority(df, class_col="class", random_state=random_state)
    if df_balanced["class"].nunique() < 2:
        return "one_class_after_balance"

    X = df_balanced[["x", "y", "z", "cluster", "dist"]]
    y = df_balanced["class"]

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        class_weight="balanced"
    )

    scoring_choice = {
        "accuracy": make_scorer(accuracy_score),
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1_score": make_scorer(f1_score, zero_division=0),
        "mcc": make_scorer(matthews_corrcoef),
    }

    scores = cross_validate(
        rf, X, y, cv=cv, scoring=scoring_choice, return_train_score=True, n_jobs=1
    )

    pd.DataFrame(scores).to_csv(out_path, index=False)
    return "ok"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trn-index", type=int, required=True, help="0-based index into TRN list")
    ap.add_argument("--trn-list-csv", default="../data/TRNs_for_classicalML.csv")
    ap.add_argument("--trn-col", default="0")
    ap.add_argument("--in-dir", default="../data/ML_outputs")
    ap.add_argument("--out-dir", default="../data/scores_outputs")
    ap.add_argument("--cv", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=50)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    trns = pd.read_csv(args.trn_list_csv)
    trn_list = trns[args.trn_col].astype(str).tolist()

    if args.trn_index < 0 or args.trn_index >= len(trn_list):
        raise SystemExit(f"trn-index {args.trn_index} out of range (0..{len(trn_list)-1})")

    trn_name = trn_list[args.trn_index]
    status = run_one_trn(
        trn_name,
        in_dir=args.in_dir,
        out_dir=args.out_dir,
        random_state=args.random_state,
        cv=args.cv,
        n_estimators=args.n_estimators,
    )
    print(f"TRN={trn_name}\tSTATUS={status}")

if __name__ == "__main__":
    main()
