#!/bin/bash

# Kaggle non supporta module e Conda. Usiamo pip per installare i pacchetti necessari.
pip install tensorflow torch scikit-learn --quiet

# Controlla se il file di input è fornito e valido
if [ -z "$1" ] || [ ! -f "$1" ]; then
    echo "Couldn't find a valid path. Using default path: data/processed/all_reviews_2017.csv"
    dataset_path="data/processed/all_reviews_2017.csv"
else
    dataset_path="$1"
fi

# Genera riassunti astrattivi
if [[ "$@" =~ "--add-padding" ]]; then
    # aggiunge l'opzione '--no-trimming' se è presente l'argomento '--add-padding'
    candidates=$(python glimpse/data_loading/generate_abstractive_candidates.py --dataset_path "$dataset_path" --scripted-run --no-trimming | tail -n 1)
else
    # esegue senza opzioni aggiuntive
    candidates=$(python glimpse/data_loading/generate_abstractive_candidates.py --dataset_path "$dataset_path" --scripted-run | tail -n 1)
fi

# Calcola i punteggi RSA basati sui riassunti generati
rsa_scores=$(python glimpse/src/compute_rsa.py --summaries $candidates | tail -n 1)

# Mostra il risultato
echo "Riassunti generati: $candidates"
echo "Punteggi RSA calcolati: $rsa_scores"
