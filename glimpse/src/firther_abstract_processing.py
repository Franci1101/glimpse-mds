import pandas as pd
import argparse

def clean_summary(input_file, output_file):
    # Carica il CSV
    df = pd.read_csv(input_file)

    # Pulisce la colonna 'summary' estraendo solo il testo
    df["summary"] = df["summary"].str.extract(r"\[list\(\['(.*)'\]\)\]")

    # Salva il file pulito
    df.to_csv(output_file, index=False)
    print(f"File pulito salvato come {output_file}")

if __name__ == "__main__":
    # Crea il parser per gli argomenti da riga di comando
    parser = argparse.ArgumentParser(description="Pulisci la colonna 'summary' di un CSV")
    parser.add_argument("input_file", type=str, help="Il file CSV di input")
    parser.add_argument("output_file", type=str, help="Il file CSV di output")

    # Parso gli argomenti
    args = parser.parse_args()

    # Chiama la funzione di pulizia
    clean_summary(args.input_file, args.output_file)
