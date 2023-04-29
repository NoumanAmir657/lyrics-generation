from imports import *
import MultiHeadAttention
import FeedForward

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        
        head_size = n_embd // n_head
        self.mha = MultiHeadAttention(n_head, head_size)
        self.ff = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.mha(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x