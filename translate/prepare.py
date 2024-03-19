import torch
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

class TranslationDataset(Dataset):
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
        src_encoded = self.encode_text(src_text, self.src_vocab, self.src_max_len)
        tgt_encoded = self.encode_text(tgt_text, self.tgt_vocab, self.tgt_max_len)
        
        return {
            "source": torch.tensor(src_encoded, dtype=torch.long),
            "target": torch.tensor(tgt_encoded, dtype=torch.long)
        }
    
    def encode_text(self, text, vocab, max_len):
        encoded = [vocab['<sos>']]  # Start-of-sequence token
        encoded += [vocab.get(word, vocab['<unk>']) for word in text.split()]  # Unknown word token
        encoded += [vocab['<eos>']]  # End-of-sequence token
        
        # Truncate or pad the sequence
        if len(encoded) > max_len:
            return encoded[:max_len]
        else:
            return encoded + [vocab['<pad>']] * (max_len - len(encoded))  # Padding token

class BPEDataset(Dataset):
    def __init__(self, sentences, vocab, max_length):
        self.sentences = sentences
        self.vocab = vocab
        self.max_length = max_length
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        encoded_sentence = tokenize_sentences([sentence], self.vocab, self.max_length)[0]
        return torch.tensor(encoded_sentence, dtype=torch.long)


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
