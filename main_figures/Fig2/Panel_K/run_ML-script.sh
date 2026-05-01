#!/bin/bash
#SBATCH --job-name=rf_trn
#SBATCH --output=logs/rf_trn_%A_%a.out
#SBATCH --error=logs/rf_trn_%A_%a.err
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --array=0-229
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=kritis@unc.edu

mkdir -p logs

python -u ML-script.py \
  --trn-index "${SLURM_ARRAY_TASK_ID}" \
  --trn-list-csv "../data/TRNs_for_classicalML.csv" \
  --trn-col "0" \
  --in-dir "../data/ML_outputs" \
  --out-dir "../data/scores_outputs" \
  --cv 3 \
  --n-estimators 50 \
  --random-state 42
