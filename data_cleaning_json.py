import json

# Load JSON data from a file
with open('data/train.jsonl', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Process each record to combine instruction and response
processed_records = []
for record in data:
    # Concatenate the instruction and the response, using a clear delimiter
    combined_text = f"Question: {record['Instruction'].strip()}\nAnswer: {record['Response'].strip()}"
    # Optionally, further cleaning can be done (removing extra spaces, etc.)
    combined_text = " ".join(combined_text.split())
    processed_records.append(combined_text)

# Write output to a text file
output_filename = "cleaned_data/general_qa_cleaned.txt"
with open(output_filename, 'w', encoding='utf-8') as outfile:
    outfile.write("\n\n".join(processed_records))

print(f"Processed data has been saved to {output_filename}")
