import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random


with open('input.txt', 'r') as f:
    text = f.read()
print(text[:50])
print(f'Length of text: {len(text)} characters')

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f'Vocabulary size: {vocab_size} characters')

itos = {i: c for i, c in enumerate(chars)}
stoi = {c: i for i, c in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(encode('Hello, world!'))
print(decode(encode('Hello, world!')))

data = torch.tensor(encode(text), dtype=torch.long)
print(data.shape, data.dtype)
train_ratio = 0.9
n = int(train_ratio * len(data))
train_data = data[:n]
val_data = data[n:]
block_size = 8
batch_size = 32

def get_batch(split, block_size=block_size , batch_size=batch_size): 
    data = train_data if split == 'train' else val_data
    ixs = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[ix:ix + block_size] for ix in ixs])
    y = torch.stack([data[ix + 1:ix + block_size + 1] for ix in ixs])
    return x, y

x, y = get_batch(train_data)

print(x.shape, y.shape)
print(x[1])
print(y[1])

@torch.no_grad()
def estimate_loss(model, eval_steps=20):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_steps)
        for k in range(eval_steps):
            x, y = get_batch(split)
            logits, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

meaner = torch.tril(torch.ones(3, 3))
meaner /= meaner.sum(dim=-1, keepdim=True)
rand = torch.ones((3, 3))
print(meaner@rand)



class BigramLanguageModel(nn.Module):
    
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
        
    def forward(self, idx, targets=None):
        logits = self.token_embedding_table(idx)
        if targets is None:
            loss = None
        else:
            b, t, c = logits.shape
            logits = logits.view(b * t, c)
            targets = targets.view(b * t)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, loss = self(idx)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

m = BigramLanguageModel(vocab_size)
logits, loss = m(x, y)
print(logits.shape)
print(loss)
print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

for i in tqdm(range(100000)):
    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % 10000 == 0:
        losses = estimate_loss(m)
        print(f'Step {i}: train loss: {losses["train"]:.4f}, val loss: {losses["val"]:.4f}')

print(decode(m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=500)[0].tolist()))