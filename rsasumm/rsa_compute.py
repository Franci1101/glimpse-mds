import argparse
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import pandas as pd
from rsasumm.rsa_reranker import RSAReranking
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/pegasus-large")
    parser.add_argument("--summaries", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to a checkpoint to resume from")
    return parser.parse_args()

def compute_rsa_scores(model, tokenizer, summaries, device):
    results = []
    grouped = list(summaries.groupby("id"))
    for name, group in tqdm(grouped):
        rsa_reranker = RSAReranking(
            model,
            tokenizer,
            device=device,
            candidates=group["summary"].unique().tolist(),
            source_texts=group["text"].unique().tolist(),
            batch_size=32,
            rationality=3
        )
        best_rsa, best_base, speaker_df, listener_df, initial_listener, language_model_proba_df, initial_consensuality_scores, consensuality_scores = rsa_reranker.rerank(t=2)
        
        results.append({
            "id": name,
            "best_rsa": best_rsa,
            "best_base": best_base,
            "speaker_df": speaker_df,
            "listener_df": listener_df,
            "initial_listener": initial_listener,
            "language_model_proba_df": language_model_proba_df,
            "initial_consensuality_scores": initial_consensuality_scores,
            "consensuality_scores": consensuality_scores
        })
    return results

def main():
    args = parse_args()

    # Carica il modello e il tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name).to(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Carica i dati dai riassunti
    summaries = pd.read_csv(args.summaries)

    # Calcola i punteggi RSA
    rsa_results = compute_rsa_scores(model, tokenizer, summaries, args.device)

    # Salva i risultati in un file CSV
    results_df = pd.DataFrame(rsa_results)
    results_df.to_csv(args.output, index=False)

    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()
