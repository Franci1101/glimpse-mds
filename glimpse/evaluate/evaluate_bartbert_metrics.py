import argparse
from pathlib import Path

import pandas as pd

from bert_score import BERTScorer

def sanitize_model_name(model_name: str) -> str:
    """
    Sanitize the model name to be used as a folder name.
    @param model_name: The model name
    @return: The sanitized model name
    """
    return model_name.replace("/", "_")

# logging.basicConfig(stream=stdout, level=logging.)
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summaries", type=Path, default="")

    # device
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args



def parse_summaries(path: Path):
    """
    :return: a pandas dataframe with at least the columns 'text' and 'summary'
    """
    # read csv file

    df = pd.read_csv(path).dropna()


    # check if the csv file has the correct columns
    if not all([col in df.columns for col in ["gold", "summary"]]):
        raise ValueError("The csv file must have the columns 'text' and 'summary'.")

    return df


def evaluate_bartbert(df, device="cuda"):
    # make a list of the tuples (text, summary)

    # texts = df.text.tolist()
    texts = df.gold.tolist()
    summaries = df.summary.tolist()

    scorer = BERTScorer(lang="en", rescale_with_baseline=True, device=device)

    metrics = {'BERTScore': []}
    for i in range(len(texts)):
        texts[i] = texts[i].replace("\n", " ")
        summaries[i] = summaries[i].replace("\n", " ")

        P, R, F1 = scorer.score([summaries[i]], [texts[i]])

        metrics['BERTScore'].append(F1.mean().item())

    # compute the mean of the metrics
    # metrics = {k: sum(v) / len(v) for k, v in metrics.items()}

    return metrics


def main():
    args = parse_args()

    path = args.summaries
    path.parent.mkdir(parents=True, exist_ok=True)
    print("inzio")
    # load the model
    df = parse_summaries(args.summaries)
    print("Input DataFrame:")
    print(df.head())
    print("Strat evaluation")
    metrics = evaluate_bartbert(df)

    # make a dataframe with the metric
    df_metrics = pd.DataFrame(metrics)

    # Add the model name in the metrics names
    df_metrics = df_metrics.add_prefix(f"common/")

    print("Metrics DataFrame:")
    print(df_metrics)

    if path.exists():
        df_old = pd.read_csv(path, index_col=0)

        # create the colums if they do not exist
        for col in df_metrics.columns:
            if col not in df.columns:
                df[col] = float("nan")

        # add entry to the dataframe
        for col in df_metrics.columns:
            df[col] = df_metrics[col]

    df.to_csv(path)
    print("Final DataFrame saved:")
    print(df)


if __name__ == "__main__":
    main()
