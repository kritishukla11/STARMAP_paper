#!/bin/bash
#SBATCH --job-name=mds_dist
#SBATCH --output=logs_mds/mds_dist_%A_%a.out
#SBATCH --error=logs_mds/mds_dist_%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --array=0-17%20
#SBATCH --mail-user=kritis@unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs_mds

GENE_LIST=gene_list.txt
COORD3D=3Dcoord_allgenes.csv
OUTPUT_DIR=output
OUTDIR=mds_distortion_chunks

python mds_distortion_chunk.py \
    --coord3d "$COORD3D" \
    --output-dir "$OUTPUT_DIR" \
    --gene-list "$GENE_LIST" \
    --outdir "$OUTDIR" \
    --chunk-id "$SLURM_ARRAY_TASK_ID" \
    --chunk-size 1000