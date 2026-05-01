#!/bin/bash
#SBATCH --job-name=prot_drug_nulls
#SBATCH --output=logs_2/prot_drug_nulls_%j.out
#SBATCH --error=logs_2/prot_drug_nulls_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G


python protein_drug_network_nulls.py \
    --input_csv drug_protein_scores.csv \
    --output_prefix protein_drug_network \
    --top_k 25 \
    --n_null 250 \
    --seed 42