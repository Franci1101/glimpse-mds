#!/bin/bash

# Kaggle non supporta module e Conda. Usiamo pip per installare pacchetti necessari
pip install tensorflow torch scikit-learn --quiet

# Controlla se il file di input è fornito e valido
if [ -z "$1" ] || [ ! -f "$1" ]; then
    echo "Couldn't find a valid path. Using default path: data/processed/all_reviews_2017.csv"
    dataset_path="data/processed/all_reviews_2017.csv"
else
    dataset_path="$1"
fi

# Genera riassunti estrattivi
candidates=$(python glimpse/data_loading/generate_extractive_candidates.py --dataset_path "$dataset_path" --scripted-run | tail -n 1)

# Calcola i punteggi RSA
rsa_scores=$(python glimpse/src/compute_rsa.py --summaries $candidates | tail -n 1)

# Mostra i risultati
echo "Generated summaries: $candidates"
echo "RSA scores: $rsa_scores"
