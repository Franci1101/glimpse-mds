import pandas as pd
import os
import re
import argparse

def clean_text(text):
    if pd.isna(text):  # If the value is NaN, return an empty string
        return ""
    
    # Keep only letters, spaces, and common punctuation
    text = re.sub(r"[^a-zA-Z0-9.,!?;:'()\s]", "", text)
    
    # Remove multiple spaces and newlines
    text = re.sub(r"\s+", " ", text).strip()
    
    return text

def process_file(input_path, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    dataset = pd.read_csv(input_path, sep=",")
    
    # Controllo se le colonne corrette esistono già
    if "text" in dataset.columns and "gold" in dataset.columns:
        sub_dataset = dataset[['id', 'text', 'gold']].copy()
    else:
        # Se non esistono, assume che siano chiamate "review" e "metareview"
        sub_dataset = dataset[['id', 'review', 'metareview']].copy()
        sub_dataset.rename(columns={"review": "text", "metareview": "gold"}, inplace=True)

    sub_dataset['text'] = sub_dataset['text'].apply(clean_text)
    sub_dataset['gold'] = sub_dataset['gold'].apply(clean_text)
    
    sub_dataset.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleans reviews in a CSV file")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file")
    parser.add_argument("output_csv", type=str, help="Path to save the cleaned CSV file")
    
    args = parser.parse_args()
    process_file(args.input_csv, args.output_csv)
