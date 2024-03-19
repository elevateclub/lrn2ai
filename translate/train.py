import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from models import Transformer  # Adjust this import
from prepare import BPEDataset, load_vocab  # Adjust these imports

src_vocab_file = 'dat/vocab-env'
tgt_vocab_file = 'dat/vocab-kor'

src_vocab = load_vocab(src_vocab_file)
tgt_vocab = load_vocab(tgt_vocab_file)

data_file = 'dat/raw.txt'
dataset = BPEDataset(data_file, src_vocab, tgt_vocab, src_max_len=50, tgt_max_len=50)

# Split dataset into training and validation sets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = Transformer(
    num_encoder_layers=3, 
    num_decoder_layers=3,
    d_model=512, 
    num_heads=8, 
    dff=2048,
    input_vocab_size=len(src_vocab),  # Use source vocab size
    target_vocab_size=len(tgt_vocab),  # Use target vocab size
    pe_input=1000, 
    pe_target=1000,
    dropout_rate=0.1
).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.0005)
loss_fn = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])

num_iterations = 10000
print_every = 100
validate_every = 1000

for iteration in range(num_iterations):
    model.train()
    total_loss = 0
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        
        src = batch['source'].to(device)
        tgt = batch['target'].to(device)
        
        # Shift the decoder input and target
        tgt_input = tgt[:, :-1]
        targets = tgt[:, 1:].contiguous().view(-1)
        
        output = model(src, tgt_input)
        output = output.view(-1, output.size(-1))
        
        loss = loss_fn(output, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
        if (batch_idx + 1) % print_every == 0:
            print(f'Iteration {iteration}, Loss: {total_loss / print_every:.4f}')
            total_loss = 0
    
    if (iteration + 1) % validate_every == 0:
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                src = batch['source'].to(device)
                tgt = batch['target'].to(device)
                
                tgt_input = tgt[:, :-1]
                targets = tgt[:, 1:].contiguous().view(-1)
                
                output = model(src, tgt_input)
                output = output.view(-1, output.size(-1))
                loss = loss_fn(output, targets)
                
                total_val_loss += loss.item()
        
        avg_val_loss = total_val_loss / len(val_loader)
        print(f'Validation Loss after {iteration + 1} iterations: {avg_val_loss:.4f}')
