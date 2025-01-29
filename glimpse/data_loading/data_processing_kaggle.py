import pandas as pd
import os
import re

def clean_text(text):
    """Pulisce il testo rimuovendo caratteri speciali, spazi multipli e simboli strani."""
    if pd.isna(text):  # Controlla se il testo è NaN
        return ""
    
    text = text.lower()  # Converti in lowercase (opzionale)
    text = re.sub(r'\s+', ' ', text)  # Rimuove spazi multipli
    text = re.sub(r'[^\w\s.,!?;:]', '', text)  # Mantiene solo lettere, numeri e punteggiatura utile
    return text.strip()  # Rimuove spazi all'inizio e alla fine

data_glimpse = "data/processed/"
os.makedirs(data_glimpse, exist_ok=True)

for year in range(2017, 2022):
    dataset = pd.read_csv(f"data/all_reviews_{year}.csv")
    
    # Seleziona le colonne e rinomina
    sub_dataset = dataset[['id', 'review', 'metareview']].copy()
    sub_dataset.rename(columns={"review": "text", "metareview": "gold"}, inplace=True)

    # Pulisce i testi
    sub_dataset["text"] = sub_dataset["text"].apply(clean_text)
    sub_dataset["gold"] = sub_dataset["gold"].apply(clean_text)

    # Filtra testi troppo corti (meno di 10 caratteri)
    sub_dataset = sub_dataset[sub_dataset["text"].str.len() > 10]
    
    # Salva il dataset pulito
    sub_dataset.to_csv(f"{data_glimpse}all_reviews_{year}.csv", index=False)

print("Preprocessing completato!")
