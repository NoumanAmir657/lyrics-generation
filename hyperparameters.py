import torch

hyperparameters = {
"batch_size":       64,
"block_size":       256,            # context length
"max_iters":        5000,           # number of iterations
"eval_interval":    500,            # evaluate on val set after every 500 iterations
"learning_rate":    3e-4,
"eval_iters":       200,            # number of loss values to mean over while estimating loss
"n_embd":           384,            # embedding size for each word
"n_head":           6,
"n_layer":          6,              # number of Blocks
"dropout":          0.2,            # for regularization
"device":           'cuda' if torch.cuda.is_available() else 'cpu'
}