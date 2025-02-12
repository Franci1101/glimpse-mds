import pickle
import pandas as pd
import argparse
import os

# Function to load the pickle file
def load_pickle(file_path):
    with open(file_path, "rb") as f:
        loaded_data = pickle.load(f)
    return loaded_data

# Main function
def process_rsa(file_path, output_csv="summaries.csv"):
    # Load the pickle file
    loaded_data = load_pickle(file_path)

    summaries = []
    if 'results' in loaded_data:
        results = loaded_data['results']
    else:
        print("Error: the file does not contain the 'results' key")
        return

    # Iterate over each document
    for result in results:
        speaker_value = result.get('speaker_df', 'Non disponibile')
        id_value = result.get('id', 'Non disponibile')
        best_rsa_value = result.get('best_rsa', 'Non disponibile')
        gold_summary = result.get('gold', 'Non disponibile')
        
        # Remove parentheses and trailing commas from the ID if it’s a tuple or strange value
        if isinstance(id_value, tuple) and len(id_value) == 1:
            id_value = id_value[0]

        # Check if best_rsa is already a list
        if not isinstance(best_rsa_value, list):
            best_rsa_value = [best_rsa_value]
        
       # Create a unique summary by concatenating the candidates
        #generated_summary = " ".join(str(sentence).replace("\n", "").replace(";", ",") for sentence in best_rsa_value)
        #generated_summary = " ".join(str(sentence).strip().replace("\n", "").replace(";", ",") for sentence in best_rsa_value)

        import re

        generated_summary = " ".join(
            re.sub(r"^\['|'\]$", "", str(sentence).strip()).replace("\n", "").replace(";", ",") 
            for sentence in best_rsa_value
        )

        
        # Pulisce la colonna 'summary' estraendo solo il testo
        generated_summary = generated_summary.extract(r"\[list\(\['(.*)'\]\)\]")

        #generated_summary = " ".join(sentence.replace("\n", "").replace(";", ",") for sentence in best_rsa_value)
        gold_summary = gold_summary.replace(";", ",")

        # Save the results for each document
        summaries.append({
            "id": id_value,
            "gold": gold_summary,
            "summary": generated_summary,
        })

    # Ensure output directory exists
    output_dir = os.path.dirname(output_csv)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Save the results in a CSV file with columns `gold` and `summary`
    df_summaries = pd.DataFrame(summaries)
    df_summaries.to_csv(output_csv, index=False)
    print(f"File CSV saved at: {output_csv}")

    return df_summaries

# Function to handle command line arguments
def main():
    parser = argparse.ArgumentParser(description="Process a pickle file and save a CSV with generated summaries.")
    parser.add_argument("input_file", type=str, help="Path to the input file pickle")
    parser.add_argument("output_file", type=str, help="Path to the output CSV file")

    args = parser.parse_args()
    process_rsa(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
