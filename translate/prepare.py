from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader

from minbpe import BasicTokenizer

class BPEDataset(Dataset):
    def __init__(self, file_path, src_vocab, tgt_vocab, src_max_len, tgt_max_len):
        self.data = []
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        # Load and preprocess data
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                id1x, str1_lang1, id1y, str1_lang2 = line.strip().split('\t')
                self.data.append((str1_lang1, str1_lang2))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text, tgt_text = self.data[idx]
        src_encoded = self.tokenize_and_encode(src_text, self.src_vocab, self.src_max_len)
        tgt_encoded = self.tokenize_and_encode(tgt_text, self.tgt_vocab, self.tgt_max_len)
        
        return {
            "source": torch.tensor(src_encoded, dtype=torch.long),
            "target": torch.tensor(tgt_encoded, dtype=torch.long)
        }
    
    def tokenize_and_encode(self, text, vocab, max_len):
        # Convert tokens to their corresponding indices in the vocabulary
        token_ids = vocab.encode(text)
        
        # Add <sos> and <eos> tokens
        token_ids = [vocab.special_tokens['<sos>']] + token_ids + [vocab.special_tokens['<eos>']]
        
        # Padding or truncation to max_len
        if len(token_ids) < max_len:
            token_ids += [vocab.special_tokens['<pad>']] * (max_len - len(token_ids))  # Padding
        else:
            token_ids = token_ids[:max_len-1] + [vocab.special_tokens['<eos>']]  # Ensure <eos> is at the end if truncating
        
        return token_ids

def load_tokenizer(f):
    t = BasicTokenizer()
    t.load(f)
    return t

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
