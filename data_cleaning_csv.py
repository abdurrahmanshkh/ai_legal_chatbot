import pandas as pd

# Load CSV data into a DataFrame
print("Loading CSV file... This may take a while for large files.")
df = pd.read_csv('data/train.csv', chunksize=10000)  # Process in chunks for efficiency

output_filename = "cleaned_data/train_cleaned.txt"
with open(output_filename, 'w', encoding='utf-8') as outfile:
    for chunk_idx, chunk in enumerate(df):
        print(f"Processing chunk {chunk_idx + 1}...")
        processed_cases = []
        for index, row in chunk.iterrows():
            # Ensure text and summary columns are strings before stripping
            text_data = str(row['Text']).strip() if pd.notna(row['Text']) else ""
            summary_data = str(row['Summary']).strip() if pd.notna(row['Summary']) else ""

            # Combine text and summary
            combined_text = f"Case Text: {text_data} \nSummary: {summary_data}"
            processed_cases.append(combined_text)
        
        # Write processed cases to file
        outfile.write("\n\n".join(processed_cases) + "\n\n")

print(f"Processing complete. Processed data has been saved to {output_filename}")
