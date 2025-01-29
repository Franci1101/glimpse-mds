from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
from tqdm import tqdm

import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from rsasumm.rsa_reranker import RSAReranking

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large-cnn")
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()

def parse_summaries(path: Path) -> pd.DataFrame:
    summaries = pd.read_csv(path)
    required_columns = {"index", "id", "id_candidate", "summary", "text", "gold"}
    
    if not required_columns.issubset(summaries.columns):
        raise ValueError(f"The CSV file must have columns: {required_columns}")

    return summaries

def compute_rsa_scores(summaries: pd.DataFrame, model, tokenizer, device):
    results = []

    for name, group in tqdm(summaries.groupby("id")):
        print(f"\nProcessing ID: {name} - Total candidates: {len(group)}")
        
        rsa_reranker = RSAReranking(
            model, tokenizer, device,
            group.summary.tolist(),
            group.text.unique().tolist()
        )

        # Calcola i punteggi RSA
        _, _, speaker_df, listener_df, initial_listener, language_model_proba_df = rsa_reranker.rerank(t=3)

        # Assegna i punteggi alle frasi candidate
        group["speaker_proba"] = group.apply(lambda row: speaker_df.loc[row["text"], row["summary"]], axis=1)
        group["listener_proba"] = group.apply(lambda row: listener_df.loc[row["text"], row["summary"]], axis=1)
        group["language_model_proba"] = group.apply(lambda row: language_model_proba_df.loc[row["text"], row["summary"]], axis=1)
        group["initial_listener_proba"] = group.apply(lambda row: initial_listener.loc[row["text"], row["summary"]], axis=1)

        results.append(group)

    # Unisci tutti i risultati e ordina le frasi dalla migliore alla peggiore
    final_results = pd.concat(results).sort_values(by="speaker_proba", ascending=False)

    return final_results

def main():
    args = parse_args()

    # Carica il modello e il tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Carica le frasi estratte
    summaries = parse_summaries(args.summaries)

    # Calcola e ordina i punteggi RSA
    ranked_summaries = compute_rsa_scores(summaries, model, tokenizer, args.device)

    # Salva il file ordinato
    output_path = Path(args.output_dir) / f"{args.summaries.stem}-rsa_ranked.csv"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ranked_summaries.to_csv(output_path, index=False)

    print(f"File salvato: {output_path}")

if __name__ == "__main__":
    main()
