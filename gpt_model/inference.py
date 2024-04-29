import numpy as np 
import matplotlib.pyplot as plt 
from glob import glob 
import os 
import random 
import pickle

import torch 
import torch.nn as nn
from torch.nn import functional as F
import sys
sys.path.append("decoder_only")
from gpt_decoder_only import *


try:
    random_seed = int(sys.argv[1])
    my_count = int(sys.argv[2])
except:
    print("please provide the value of seed and count")
    exit(0)


torch_seed = random_seed
torch.manual_seed(torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(torch_seed)


random.seed(torch_seed)
np.random.seed(torch_seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)


batch_size = 128
block_size = 128

max_iters = 10000
eval_iters = 10

learning_rate = 1e-4
n_embd = 256
n_head = 8
n_layer = 8
dropout = 0.2


vocab = np.loadtxt("vocab.txt")
vocab_size = len(vocab)


model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, device).to(device)
print('loading model parameters...')
with open(f'model_best_3state_latent_2d_run_{my_count}.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully!')



prompt = np.loadtxt("inference_data.txt")
prompt = prompt[:200]
context = torch.tensor(prompt, dtype=torch.long, device=device)
generated_chars = model.generate(context.unsqueeze(0), block_size = block_size, max_new_tokens=200000)[0].tolist()
np.savetxt(f"generated_state_from_gpt_3state_latent_2d_run_{my_count}.txt", np.array([generated_chars]).T, fmt = "%d")

