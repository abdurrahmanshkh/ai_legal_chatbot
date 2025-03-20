import re

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove newline characters and extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove any unwanted characters (keeping letters, numbers, basic punctuation)
    text = re.sub(r'[^a-z0-9.,;:?!\'"()\-\s]', '', text)
    # Trim leading/trailing spaces
    return text.strip()

def process_file(input_file, output_file):
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            text = infile.read()
        
        cleaned_text = clean_text(text)
        
        with open(output_file, 'w', encoding='utf-8') as outfile:
            outfile.write(cleaned_text)
        
        print(f"Cleaned text has been saved to {output_file}")
    except Exception as e:
        print(f"Error: {e}")

# Example usage
input_filename = "input.txt"  # Change to your input file
output_filename = "output.txt"  # Change to your desired output file
process_file(input_filename, output_filename)
