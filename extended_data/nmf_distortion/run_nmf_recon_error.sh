#!/bin/bash
#SBATCH --job-name=nmf_recon_chunk
#SBATCH --output=logs/nmf_recon_chunk_%A_%a.out
#SBATCH --error=logs/nmf_recon_chunk_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-17%20
#SBATCH --mail-user=kritis@unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs

GENE_LIST=gene_list.txt
INFO_CSV=3Dcoord_allgenes.csv
OUTDIR=reconstruction_error/per_chunk

python nmf_recon_error.py \
    --gene-list "$GENE_LIST" \
    --info-csv "$INFO_CSV" \
    --outdir "$OUTDIR" \
    --chunk-id "$SLURM_ARRAY_TASK_ID" \
    --chunk-size 1000 \
    --k-min 3 \
    --k-max 6 \
    --max-iter 200