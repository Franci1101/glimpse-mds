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
        for idx, summary in group.iterrows():
            summary_text = summary['summary']
            rsa_value = speaker_df.loc[summary_text] if summary_text in speaker_df.index else None  # Prende RSA per ogni frase

            row = {
                "id": name,
                "id_candidate": summary['id_candidate'],
                "summary": summary_text,
                "rsa": rsa_value,  # Valore RSA per ogni frase
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
