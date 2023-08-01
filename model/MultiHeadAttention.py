from imports import *
import Head

# contains mutiple heads that are executed in parallel
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        
        # projects to n_embd
        self.linear = nn.Linear(head_size * num_heads, hp['n_embd'])
        
        self.dropout = nn.Dropout(hp['dropout'])

    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim=1)
        out = self.linear(out)      # (B, T, n_embd)
        out = self.dropout(out)     # (B, T, n_embd)
        
        return out