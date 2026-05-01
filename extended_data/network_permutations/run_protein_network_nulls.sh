#!/bin/bash
#SBATCH --job-name=prot_nulls
#SBATCH --output=logs_2/prot_nulls_%j.out
#SBATCH --error=logs_2/prot_nulls_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G

mkdir -p logs_2

python protein_network_nulls.py \
    --input_csv top_protein_pathway_combos.csv \
    --output_prefix protein_network \
    --top_k 25 \
    --n_null 250 \
    --seed 42