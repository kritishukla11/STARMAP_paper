#!/bin/bash
#SBATCH --job-name=protein_mlp_r2
#SBATCH --output=logs_r2/protein_mlp_r2_%j.out
#SBATCH --error=logs_r2/protein_mlp_r2_%j.err
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --mail-user=kritis@unc.edu
#SBATCH --mail-type=BEGIN,END,FAIL

mkdir -p logs_r2

python mlp_r2_script-noml.py \
    --input_dir compare_outputs_noml \
    --output_csv protein_mlp_r2_noml.csv