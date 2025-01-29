import pickle
from pathlib import Path
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, PegasusTokenizer
import argparse
from tqdm import tqdm
import sys, os.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from rsasumm.rsa_reranker import RSAReranking


DESC = """
Compute the RSA matrices for all the set of multi-document samples and dump these along with additional information in a CSV file.
"""

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="google/pegasus-arxiv")
    parser.add_argument("--summaries", type=Path, default="")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--filter", type=str, default=None)
    parser.add_argument("--scripted-run", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", type=str, default="cuda")

    return parser.parse_args()


def parse_summaries(path: Path) -> pd.DataFrame:
    try:
        summaries = pd.read_csv(path)
    except:
        raise ValueError(f"Unknown dataset {path}")

    # check if the dataframe has the right columns
    if not all(
        col in summaries.columns for col in ["index", "id", "text", "gold", "summary", "id_candidate"]
    ):
        raise ValueError(
            "The dataframe must have columns ['index', 'id', 'text', 'gold', 'summary', 'id_candidate']"
        )

    return summaries


def compute_rsa(summaries: pd.DataFrame, model, tokenizer, device):
    rows = []
    for name, group in tqdm(summaries.groupby(["id"])):
        rsa_reranker = RSAReranking(
            model,
            tokenizer,
            device=device,
            candidates=group.summary.tolist(),  # Usa tutte le frasi candidate
            source_texts=group.text.unique().tolist(),
            batch_size=16,
            rationality=3,
        )
        (
            best_rsa,
            best_base,
            speaker_df,
            listener_df,
            initial_listener,
            language_model_proba_df,
            initial_consensuality_scores,
            consensuality_scores,
        ) = rsa_reranker.rerank(t=2)

        gold = group['gold'].tolist()[0]

        # Ora salva TUTTE le frasi candidate con il loro RSA
        def normalize_text(text):
            """Normalizza il testo: lowercase, rimozione caratteri speciali e normalizzazione Unicode"""
            text = text.lower().strip()  # Lowercase e rimuove spazi extra
            text = unicodedata.normalize("NFKD", text)  # Normalizza caratteri Unicode
            text = re.sub(r"\s+", " ", text)  # Rimuove spazi multipli
            text = re.sub(r"[^\w\s]", "", text)  # Rimuove punteggiatura e simboli strani
            return text
        
        for idx, summary in group.iterrows():
            summary_text = normalize_text(summary['summary'])  # Normalizza la frase
        
            # Normalizza anche le chiavi di speaker_df
            speaker_df.index = speaker_df.index.map(normalize_text)
        
            # DEBUG: Controlliamo se la frase normalizzata è presente
            print(f"\nProcessing summary: {summary_text}")
            print(f"Keys in speaker_df (normalized): {list(speaker_df.index)[:5]} ...")
        
            if summary_text in speaker_df.index:
                rsa_value = speaker_df.loc[summary_text]
            else:
                print(f"⚠️ Warning: '{summary_text}' not found in speaker_df!")
                rsa_value = None  # Evita errori
        
            row = {
                "id": name,
                "id_candidate": summary['id_candidate'],
                "summary": summary['summary'],  # Manteniamo l'originale
                "rsa": rsa_value,  # Valore RSA
                "gold": gold,
            }
            rows.append(row)

    return rows


def main():
    args = parse_args()

    if args.filter is not None:
        if args.filter not in args.summaries.stem:
            return

    # load the model and the tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_name)
    if "pegasus" in args.model_name: 
        tokenizer = PegasusTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    model = model.to(args.device)

    # load the summaries
    summaries = parse_summaries(args.summaries)

    # rerank the summaries and collect the results
    rows = compute_rsa(summaries, model, tokenizer, args.device)

    # Verifica se la directory di output esiste, altrimenti creala
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Salva i risultati in un file CSV
    output_csv_path = Path(args.output_dir) / f"{args.summaries.stem}-rsa_results.csv"
    df = pd.DataFrame(rows)

    # Salva nel file CSV
    df.to_csv(output_csv_path, index=False)

    print(f"CSV saved at: {output_csv_path}")


if __name__ == "__main__":
    main()
