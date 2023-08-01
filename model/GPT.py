from imports import *
import Block

class GPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.vocab_embedding        = nn.Embedding(vocab_size, hp['n_embd'])
        self.positional_embedding   = nn.Embedding(hp['block_size'], hp['n_embd'])
        self.blocks                 = nn.Sequential(*[Block(hp['n_embd'], hp['n_head']) for _ in range(hp['n_layer'])])
        self.lnf                    = nn.LayerNorm(hp['n_embd'])
        self.linear                 = nn.Linear(hp['n_embd'], vocab_size)

        # used for weights initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x, y=None):
        # x.shape --> (B, T)
        # y.shape --> (B, T)
        B, T = x.shape

        vocab_emb = self.vocab_embedding(x)
        pos_emb = self.positional_embedding(torch.arange(T, device=hp['device']))
        x = vocab_emb + pos_emb     # (B, T, C)
        x = self.blocks(x)          # (B, T, C)
        x = self.lnf(x)
        out = self.linear(x)        # (B, T, vocab_size)

        if y is None:
            loss = None
        else:
            B, T, C = out.shape
            out = out.view(B*T, C)
            y = y.view(B*T)
            loss = F.cross_entropy(out, y)

        return out, loss
    
    def generate(self, x, max_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_tokens):
            # crop idx to the last block_size tokens
            x_cond = x[:, -hp['block_size']:]
            # get the predictions
            logits, _ = self(x_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            x_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            x = torch.cat((x, x_next), dim=1) # (B, T+1)
        return x