import re
import torch

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

def generate_bigrams(words, N, stoi):
    window_size = 1
    words_with_dot_token = (['.'] * window_size) + words + ['.']
    words_temp = words_with_dot_token[window_size:]

    for i,w in enumerate(words_temp):
        i += window_size
        context = words_with_dot_token[i-window_size:i]
        target = w
    
        N[stoi[context[0]], stoi[target]] += 1

def get_loss(words, P):
    window_size = 1
    words_with_dot_token = (['.'] * window_size) + words + ['.']
    words_temp = words_with_dot_token[window_size:]
    
    n = 0
    nlls = 0
    for i,w in enumerate(words_temp):
        i += window_size
        context = words_with_dot_token[i-window_size:i]
        target = w

        probability = P[stoi[context[0]], stoi[target]]
        log_probability = torch.log(probability)
        nll = -log_probability
        nlls += nll
        n += 1
    return (nlls / n).item()

def generate_lyrics(P, itos):
    g = torch.Generator().manual_seed(2147483647)
    for _ in range(5):
        out = []
        ix = 0
        while True:
            p = P[ix]
            ix = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
            out.append(itos[ix])
            if ix == 0:
                break
    print(' '.join(out) + '\n\n\n')

text, words, vocab, vocab_size = read_and_clean_dataset("lyrics/lyrics.txt")

stoi = create_stoi_mapping(vocab)
itos = create_itos_mapping(stoi)

N = torch.zeros((vocab_size, vocab_size), dtype=torch.int32)

generate_bigrams(words, N, stoi)

P = (N+1).float()
P /= P.sum(1, keepdims=True)

loss = get_loss(words, P)
print(f'NLL = {loss:.8f}')

generate_lyrics(P, itos)
