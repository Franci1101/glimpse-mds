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
    sub_dataset = dataset[['id', 'review', 'metareview']].copy()

    sub_dataset['review'] = sub_dataset['review'].apply(clean_text)
    sub_dataset['metareview'] = sub_dataset['metareview'].apply(clean_text)

    sub_dataset.rename(columns={"review": "text", "metareview": "gold"}, inplace=True)
    
    sub_dataset.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cleans reviews in a CSV file")
    parser.add_argument("input_csv", type=str, help="Path to the input CSV file")
    parser.add_argument("output_csv", type=str, help="Path to save the cleaned CSV file")
    
    args = parser.parse_args()
    process_file(args.input_csv, args.output_csv)
