import pandas as pd
import argparse

# Function to load the CSV file
def load_csv(file_path):
    return pd.read_csv(file_path)

# Main function
def calculate_rouge_and_bert(file_path):
    # Upload CSV file
    df = load_csv(file_path)

    # Compute average ROUGE
    global_rouge_scores = df[["common/rouge1", "common/rouge2", "common/rougeL", "common/rougeLsum"]].mean()
    print("Average common:")
    print(round(global_rouge_scores*100, 2))

    df["rouge_mean_per_document"] = df[["common/rouge1", "common/rouge2", "common/rougeL", "common/rougeLsum"]].mean(axis=1)

    global_rouge_mean = df["rouge_mean_per_document"].mean()
    print(f"Global ROUGE score: {global_rouge_mean*100:.2f}%")

    bert_scores = df['common/BERTScore']

    average_bert_score = bert_scores.mean()
    print(f"Average BERTScore: {average_bert_score*100:.2f}%")

# Function to parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Compute ROUGE and BERTScore metrics from a CSV file.")
    parser.add_argument('input_file', type=str, help="Path to the input CSV file")
    return parser.parse_args()

# main function to execute the code
def main():
    args = parse_args()
    calculate_rouge_and_bert(args.input_file)

if __name__ == "__main__":
    main()
