from imports import *

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        
        self.seq = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(hp['dropout'])
        )

    def forward(self, x):
        return self.seq(x)