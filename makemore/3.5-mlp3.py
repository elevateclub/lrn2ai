import torch
import torch.nn.functional as F
import nn
import matplotlib.pyplot as plt

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# read words
words = open('names.txt', 'r').read().splitlines()
len(words)

# build vocab
chars = sorted(list(set(''.join(words))))
stoi = {s:i+1 for i,s in enumerate(chars)}
stoi['.'] = 0
itos = {i:s for s,i in stoi.items()}
vocab_size = len(itos)

# build dataset
block_size = 3 # context length: how many chars do we take to predict the next one?

def build_dataset(words):
    X, Y = [], []

    for w in words:
        context = [0] * block_size
        for ch in w + '.':
            ix = stoi[ch]
            X.append(context)
            Y.append(ix)
            context = context[1:] + [ix] # crop and append
        
    X = torch.tensor(X).to(dev)
    Y = torch.tensor(Y).to(dev)
    return X, Y

import random
random.seed(42)
random.shuffle(words)
n1 = int(0.8*len(words))
n2 = int(0.9*len(words))

Xtr, Ytr = build_dataset(words[:n1])
Xdev, Ydev = build_dataset(words[n1:n2])
Xte, Yte = build_dataset(words[n2:])

# MLP
n_embd = 10 # dimensionality of the character embedding vectors
n_hidden = 200 # number of neurons in the hidden layer of MLP

g = torch.Generator().manual_seed(2147483647)
C = torch.randn((vocab_size, n_embd),             generator=g).to(dev)
layers = [
    nn.Linear(n_embd * block_size, n_hidden, device=dev), nn.BatchNorm1d(n_hidden, device=dev), nn.Tanh(),
    nn.Linear(          n_hidden, n_hidden, device=dev), nn.BatchNorm1d(n_hidden, device=dev), nn.Tanh(),
    nn.Linear(          n_hidden, n_hidden, device=dev), nn.BatchNorm1d(n_hidden, device=dev), nn.Tanh(),
    nn.Linear(          n_hidden, n_hidden, device=dev), nn.BatchNorm1d(n_hidden, device=dev), nn.Tanh(),
    nn.Linear(          n_hidden, n_hidden, device=dev), nn.BatchNorm1d(n_hidden, device=dev), nn.Tanh(),
    nn.Linear(          n_hidden, vocab_size, device=dev), nn.BatchNorm1d(vocab_size, device=dev)
]

with torch.no_grad():
    # last layer: make less confident
    layers[-1].gamma *= 0.1
    # all other layers: apply gain
    for layer in layers[:-1]:
        if isinstance(layer, nn.Linear):
            layer.weight *= 5/3

parameters = [C] + [p for layer in layers for p in layer.parameters()]
print(sum(p.nelement() for p in parameters)) # num parameters
for p in parameters:
    p.requires_grad = True

# Optimization
max_steps = 10000
batch_size = 32
lossi = []

for i in range(max_steps):
    # minibatch
    ix = torch.randint(0, Xtr.shape[0], (batch_size,), generator=g)
    Xb, Yb = Xtr[ix], Ytr[ix] # batch X, Y

    # forward pass
    emb = C[Xb] # embed the characters into vectors
    x = emb.view(emb.shape[0], -1) 
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, Yb) # loss

    # backward pass
    for p in parameters:
        p.grad = None
    loss.backward()

    # update
    lr = 0.1 if i < 100000 else 0.01 # step learning rate decay
    for p in parameters:
        p.data += -lr * p.grad

    # track stats
    if i % 100 == 0: # print every once in a while
        print(f'{i:7d}/{max_steps:7d}: {loss.item():.4f}')
    lossi.append(loss.log10().item())    

"""
# visualize histograms
plt.figure(figsize=(20,4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, nn.Tanh):
        t = layer.out
        print('layer %d (%10s): mean %+.2f, std %.2f satuated: %.2f%%' % (i, layer.__class__.__name__, t.mean(), t.std(), 0))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__})')
plt.legend(legends)
plt.title('activation distribution')

plt.figure(figsize=(20,4))
legends = []
for i, layer in enumerate(layers[:-1]):
    if isinstance(layer, nn.Tanh):
        t = layer.out.grad
        print('layer %d (%10s): mean %+f, std %e' % (i, layer.__class__.__name__, t.mean(), t.std()))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'layer {i} ({layer.__class__.__name__}')
plt.legend(legends);
plt.title('gradient distribution')

plt.figure(figsize=(20, 4))
legends = []
for i, p in enumerate(parameters):
    t = p.grad
    if p.ndim == 2:
        print('weight %10s | mean %+f | std %e | grad:data ratio %e' % (tuple(p.shape), t.mean(), t.std(), t.std() / p.std()))
        hy, hx = torch.histogram(t, density=True)
        plt.plot(hx[:-1].detach(), hy.detach())
        legends.append(f'{i} {tuple(p.shape)}')
plt.legend(legends)
plt.title('weights gradient distribution')
"""

# calculate loss
@torch.no_grad()
def split_loss(split):
    xd,y = {
        'train': (Xtr, Ytr),
        'val': (Xdev, Ydev),
        'test': (Xte, Yte),
    }[split]
    emb = C[xd] # (N, block_size, n_embd)
    x = emb.view(emb.shape[0], -1) 
    for layer in layers:
        x = layer(x)
    loss = F.cross_entropy(x, y) # loss
    print(split, loss.item())

# put layers into eval mode
for layer in layers:
    layer.training = False
split_loss('train')
split_loss('val')

# sample from the model
g = torch.Generator().manual_seed(2147483647 + 10)

for _ in range(20):
    out = []
    context = [0] * block_size
    while True:
        # forward pass the neural net
        emb = C[torch.tensor([context])] # (1,block_size,n_embd)
        x = emb.view(emb.shape[0], -1) 
        for layer in layers:
            x = layer(x)
        logits = x
        probs = F.softmax(logits, dim=1)
        # sample from the distribution
        ix = torch.multinomial(probs, num_samples=1, generator=g).item()
        # shift the content window and track the sample
        context = context[1:] + [ix]
        out.append(ix)  
        # if we sample the special '.' token, break
        if ix == 0:
            break
    
    print(''.join(itos[i] for i in out))