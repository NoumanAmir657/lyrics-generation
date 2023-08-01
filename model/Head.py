from imports import *

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        
        self.key    =   nn.Linear(hp['n_embd'], head_size, bias=False)
        self.query  =   nn.Linear(hp['n_embd'], head_size, bias=False)
        self.val    =   nn.Linear(hp['n_embd'], head_size, bias=False)

        self.register_buffer("tril", torch.tril(torch.ones(hp['block_size'], hp['block_size'])))

        self.dropout = nn.Dropout(hp['dropout'])

    def forward(self, x):
        # input shape --> (batch, time_step, channels)
        B, T, C = x.shape
        k = self.key(x)     # (B, T, head_size)
        q = self.query(x)   # (B, T, head_size)

        # affinities
        rel = q @ k.transpose(-2, -1)   # (B, T, T)
        rel = rel * k.shape[-1]**-0.5   # (B, T, T)

        # assigns -inf to future contexts
        rel = rel.masked_fill(self.tril[:T, :T] == 0, float('-inf'))    # (B, T, T)
        rel = F.softmax(rel, dim=-1)    # (B, T, T)
        rel = self.dropout(rel)         # (B, T, T)

        # value
        v = self.value(x)   # (B, T, head_size)
        out = rel @ v       # (B, T, head_size)

        return out