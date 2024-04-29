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




torch_seed = 42
torch.manual_seed(torch_seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(torch_seed)


random_seed = 42
random.seed(random_seed)

np_seed = 42
np.random.seed(np_seed)

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
print(vocab_size)

def get_random_chunk(split):
    filename = "train_data.txt" if split == 'train' else "val_data.txt"
    file = np.loadtxt(filename)
    file_size = len(file)
    start_pos = random.randint(0, (file_size) - block_size*batch_size)
    # print("start_pos = ", start_pos)
    chunk = np.loadtxt(filename, skiprows=start_pos, max_rows=(batch_size * block_size)-1)
    data = torch.tensor(chunk, dtype=torch.long)
    return data
    
def get_batch(split):
    data = get_random_chunk(split)
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out



model = GPTLanguageModel(vocab_size, n_embd, block_size, n_layer, n_head, device).to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


train_loss = []
val_loss = []
epochs = []
best_loss = float('inf')

for iter in range(max_iters):
    # print(iter)
    if iter % eval_iters == 0:
        losses = estimate_loss()
        train_loss.append(losses['train'].item())
        val_loss.append(losses['val'].item())
        epochs.append(iter)
        print(f"step: {iter}, train loss: {losses['train']:.3f}, val loss: {losses['val']:.3f}")
        if(losses['val'].item()<best_loss):
            best_loss = losses['val'].item()
            with open(f'model_best_3state_latent_2d_run_1.pkl', 'wb') as f:
                pickle.dump(model, f)

    xb, yb = get_batch('train')

    logits, loss = model.forward(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()



with open(f'model_3state_latent_2d_run_1.pkl', 'wb') as f:
    pickle.dump(model, f)
print('model saved')


train_loss = np.array(train_loss)
val_loss = np.array(val_loss)
epochs = np.array(epochs)
np.savetxt("train_val_loss_3state_latent_2d_run_1.txt", np.array([epochs, train_loss, val_loss]).T, fmt = "%0.3e", delimiter = "\t")