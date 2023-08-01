import sys
sys.path.insert(0, './model')

from imports import *
import re
import GPT

torch.manual_seed(1337)

with open('lyrics/dataset.txt', 'r', encoding='utf-8') as f:
    text = f.read()[:200]
    text = text.lower().replace('\n', ' \n ')
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'[0-9]', '', text)
    text = re.sub(r'_', '', text)
    words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']

# unique words
vocab = sorted(list(set((words))))
vocab_size = len(vocab)
print(f'Vocabulary_size = {vocab_size}')

stoi = {s:i for i,s in enumerate(vocab)}
itos = {i:s for s,i in stoi.items()}

encode = lambda s: [stoi[c] for c in s if c in vocab]
decode = lambda l: ''.join([itos[i] for i in l])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # first 90% will be train, rest val
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - hp['block_size'], (hp['batch_size'],))
    x = torch.stack([data[i:i+hp['block_size']] for i in ix])
    y = torch.stack([data[i+1:i+hp['block_size']+1] for i in ix])
    x, y = x.to(hp['device']), y.to(hp['device'])
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(hp['eval_iters'])
        for k in range(hp['eval_iters']):
            X, Y = get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(vocab_size)
m = model.to(hp['device'])
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=hp['learning_rate'])

for iter in range(hp['max_iters']):

    # every once in a while evaluate the loss on train and val sets
    if iter % hp['eval_interval'] == 0 or iter == hp['max_iters'] - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=hp['device'])
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))