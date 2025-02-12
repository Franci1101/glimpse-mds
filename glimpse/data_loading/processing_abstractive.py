import pandas as pd
import argparse

def concat_text_by_id(input_csv, output_csv):
    # Legge il file CSV
    df = pd.read_csv(input_csv)
        
    # Raggruppa per 'id', mantiene il primo valore di 'gold' e concatena i testi
    grouped = df.groupby('id').agg({'gold': 'first', 'text': ' '.join}).reset_index()
    
    # Salva il risultato in un nuovo file CSV
    grouped.to_csv(output_csv, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Concatena i testi per ID in un file CSV.")
    parser.add_argument("input_csv", help="Percorso del file CSV di input")
    parser.add_argument("output_csv", help="Percorso del file CSV di output")
    
    args = parser.parse_args()
    concat_text_by_id(args.input_csv, args.output_csv)
