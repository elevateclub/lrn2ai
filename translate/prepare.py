def process_tsv(input_file, src_output_file, tgt_output_file):
    """
    Process a TSV file containing parallel sentences in two languages.
    The input file should have the format: id1\tstr1-lang1\tid2\tstr1-lang2
    
    Args:
    - input_file (str): Path to the input TSV file.
    - src_output_file (str): Path to save the processed source language sentences.
    - tgt_output_file (str): Path to save the processed target language sentences.
    """
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(src_output_file, 'w', encoding='utf-8') as src_out, \
         open(tgt_output_file, 'w', encoding='utf-8') as tgt_out:
        
        for line in infile:
            parts = line.strip().split('\t')
            if len(parts) != 4:
                print(f"Skipping malformed line: {line.strip()}")
                continue
            
            _, src_sentence, _, tgt_sentence = parts
            
            # Write the source and target sentences to their respective files
            src_out.write(src_sentence + '\n')
            tgt_out.write(tgt_sentence + '\n')

if __name__ == "__main__":
    # Example usage
    input_file = 'dat/raw.txt'
    src_output_file = 'dat/eng.txt'
    tgt_output_file = 'dat/kor.txt'
    
    process_tsv(input_file, src_output_file, tgt_output_file)
