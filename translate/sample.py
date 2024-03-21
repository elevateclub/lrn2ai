# Hyperparams
max_new_tokens = 16
temperature = 1.0
# top_k = 200
num_samples = 3

from contextlib import nullcontext
import os

import torch

import models
import prepare

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
out_dir = 'out' # -v0.5'
ckpt_path = os.path.join(out_dir, 'model_checkpoint_10.pt')
checkpoint = torch.load(ckpt_path, map_location=device)

# Vocab + Model
src_vocab_file = 'dat/vocab-eng/basic.model'
tgt_vocab_file = 'dat/vocab-kor/basic.model'

src_vocab = prepare.load_tokenizer(src_vocab_file)
tgt_vocab = prepare.load_tokenizer(tgt_vocab_file)

model = models.Transformer(
    num_encoder_layers=3, 
    num_decoder_layers=3,
    d_model=32, 
    num_heads=16, 
    dff=2048,
    input_vocab_size=len(src_vocab),  # Use source vocab size
    target_vocab_size=len(tgt_vocab),  # Use target vocab size
    pe_input=1000, 
    pe_target=1000,
    dropout_rate=0.1
)

# Load
state_dict = checkpoint['model_state_dict']
model.load_state_dict(state_dict)
model.eval()
model.to(device)
model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Prompt
prompt = 'hello' # input("> ")
start_ids = torch.tensor(prepare.tokenize_and_encode(prompt, src_vocab, 16), dtype=torch.long, device=device).unsqueeze(0)
out_ids = torch.tensor([models.special_tokens['<sos>']], dtype=torch.long, device=device).unsqueeze(0)
# x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

# Sample
for k in range(num_samples):
    y = model.generate(start_ids, out_ids, max_new_tokens)
    print(tgt_vocab.decode(y[0].tolist()))
    print('---------------')