import pickle
import pandas as pd
import re
import argparse

# Function to clean text
def clean_text(text):
    text = re.sub(r"[^\w\s.,!?;:'()]", "", text)  # Keep letters, numbers, and specific punctuation
    text = re.sub(r"\s+", " ", text).strip()  # Remove newlines and normalize spaces
    return text

# Function to process the pickle file and create the output CSV
def process_file(input_file, output_file):
    # Load the pickle file
    with open(input_file, "rb") as f:
        results = pickle.load(f)

    data = []

    for result in results['results']:
        id_value = result['id']

        # If id is a tuple with one element, extract it
        if isinstance(id_value, tuple) and len(id_value) == 1:
            id_value = id_value[0]  

        gold_text = result.get('gold', '')  # Default to empty string if 'gold' is missing
        
        # Flatten and remove duplicates from best_rsa sentences
        best_rsa_sentences = sum(result['best_rsa'], [])  
        seen = set()
        unique_sentences = [s for s in best_rsa_sentences if not (s in seen or seen.add(s))]

        num_final_sentences = len(unique_sentences)
        summary_text = clean_text(" ".join(unique_sentences))
        
        data.append([id_value, gold_text, summary_text])

    # Create DataFrame and save to CSV
    df = pd.DataFrame(data, columns=["id", "gold", "summary"])
    df.to_csv(output_file, index=False)

    print(f"CSV file saved: {output_file}")

# Parse command-line arguments
def main():
    parser = argparse.ArgumentParser(description="Process RSA extractive results and save them to a CSV file.")
    parser.add_argument("input_file", type=str, help="Path to the input pickle file")
    parser.add_argument("output_file", type=str, help="Path to save the output CSV file")

    args = parser.parse_args()

    process_file(args.input_file, args.output_file)

if __name__ == "__main__":
    main()
