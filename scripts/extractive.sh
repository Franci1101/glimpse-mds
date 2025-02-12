#!/bin/bash

# Kaggle non supporta module e Conda. Usiamo pip per installare i pacchetti necessari.
pip install tensorflow torch scikit-learn --quiet


# Check if input file path is provided and valid
if [ -z "$1" ] || [ ! -f "$1" ]; then
    # if no path is provided, or the path is invalid, use the default test dataset
    echo "Couldn't find a valid path. Using default path: data/processed/all_reviews_2017.csv"
    dataset_path="data/processed/all_reviews_2017.csv"
else
    dataset_path="$1"
fi

# Generate extractive summaries
candidates=$(python glimpse/data_loading/generate_extractive_candidates.py --dataset_path "$dataset_path" --scripted-run | tail -n 1)

# Compute the RSA scores based on the generated summaries
rsa_scores=$(python glimpse/src/compute_rsa.py --summaries $candidates | tail -n 1)

