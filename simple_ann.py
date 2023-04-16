import re
import torch
import torch.nn.functional as F

def read_and_clean_dataset(dataset_path):
    with open(dataset_path) as f:
        text = f.read()

    text = text.lower().replace('\n', ' \n ')
    text = re.sub(r'[^\w\s]', '', text)
    words = [w for w in text.split(' ') if w.strip() != '' or w == '\n']

    vocab = sorted(list(set(words + ['.'])))
    vocab[0] = '.'
    vocab[1] = '\n'
    vocab_size = len(vocab)

    return text, words, vocab, vocab_size

def create_stoi_mapping(vocab):
    return {s:i for i,s in enumerate(vocab)}

def create_itos_mapping(stoi):
   return {i:s for s,i in stoi.items()}

def generate_inputs_targets(words, stoi):
    window_size = 1
    words_with_dot_token = (['.'] * window_size) + words + ['.']
    words_temp = words_with_dot_token[window_size:]
    
    inputs = []
    outputs = []
    for i,w in enumerate(words_temp):
        i += window_size
        inputs.append(stoi[words[i-window_size:i][0]])
        outputs.append(stoi[w])

    return torch.tensor(inputs), torch.tensor(outputs)

def generate_lyrics(W):
    g = torch.Generator().manual_seed(2147483647)

    for _ in range(1):
        out = []
        ix = 0
        while True:
            x_one_hot = F.one_hot(torch.tensor([ix]), num_classes=vocab_size).float()
            logits = x_one_hot @ W
            counts = logits.exp()
            p = counts / counts.sum(1, keepdims=True)
        
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
            print(' '.join(out) + '\n')


text, words, vocab, vocab_size = read_and_clean_dataset("lyrics.txt")

stoi = create_stoi_mapping(vocab)
itos = create_itos_mapping(stoi)

x, y = generate_inputs_targets(words, stoi)

x_one_hot = F.one_hot(x, num_classes=vocab_size).float()

g = torch.Generator().manual_seed(2147483647)
W = torch.randn((vocab_size, vocab_size), generator=g, requires_grad=True)

for k in range(1000):  
  logits = x_one_hot @ W
  counts = logits.exp()
  probs = counts / counts.sum(1, keepdims=True)
  loss = -probs[torch.arange(x.shape[0]), y].log().mean() + 0.01*(W**2).mean()

  if (k % 100) == 0:
    print(loss.item())
  
  W.grad = None
  loss.backward()
  
  W.data += -50 * W.grad

generate_lyrics(W)