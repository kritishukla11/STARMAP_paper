#!/bin/bash
#SBATCH --job-name=logrank_scan
#SBATCH --output=logs/logrank_scan_%A_%a.out
#SBATCH --error=logs/logrank_scan_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19

mkdir -p logs
mkdir -p logrank_scan_outputs
mkdir -p logrank_scan_outputs/summary
mkdir -p logrank_scan_outputs/per_gene_days

python gene_logrank_scan.py \
    --structure_df structure_df_TCGA.csv \
    --survival_df survival_data.csv \
    --outdir logrank_scan_outputs \
    --task_id ${SLURM_ARRAY_TASK_ID} \
    --n_tasks 20 \
    --min_day 365 \
    --min_group_size 5 \
    --step 90