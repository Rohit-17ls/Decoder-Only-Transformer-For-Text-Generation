import torch
import torch.nn as nn
from torch.nn import functional as F

import random
import re

# Hyperparameters
batch_size = 64   # B
block_size = 256  # T (Context length)
n_layers = 6      # Number of blocks or units of the decoder in the architecture
n_embd = 384      # num_heads * head_size
num_heads = 6     # Number of attention heads in multiheaded attention
head_size = n_embd // num_heads  # Sequence length processed by a single head of attention
max_iters = 5000
eval_iters = 200
eval_interval = 500
learning_rate = 3e-4
dropout = 0.2      # % dropout
device = 'cuda' if torch.cuda.is_available() else 'cpu'




# One head of self attention in multiheaded attentino
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()  # Initialize parameters for the derived class object
        
        self.head_size = head_size
        self.query = nn.Linear(n_embd, head_size, bias = False) # Part of sequence being processes currently
        self.key = nn.Linear(n_embd, head_size, bias = False)   # Parts of the sequence to attend to
        self.value = nn.Linear(n_embd, head_size, bias = False) # Parts of sequence other than the current part
        # tril will have no learnable parameters
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) 
        
        self.dropout = nn.Dropout(dropout)
        
    # Forward method of a derived class of nn.Module is called by the __call__ method of the base nn.Module
    # class. Objects of classes with __call__ method are called 'callable objects'
    def forward(self, x):
        B, T, C = x.shape # B - batch_size, T - block_size (time dimension), C - channels
        k = self.key(x) # (B, T, head_size)
        q = self.key(x) # (B, T, h_s)
        wei = (q @ k.transpose(-2, -1)) * (self.head_size**(-0.5)) # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)
        
        # Weighted aggregation of values
        v = self.value(x)  # (B, T, h_s)
        out = wei @ v      # (B, T, h_s)
        return out;
        
        
# Multiple heads of attention
class MultiHeadedAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        out = torch.cat([head(x) for head in self.heads], dim = -1)
        out = self.dropout(self.proj(out))
        return out
        
        
# Feed Forward Neural Network
class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        out = self.network(x)
        return out
    

# A block in the transformer decoder
class Block(nn.Module):
    def __init__(self, n_embd, num_heads):
        super().__init__()
        self.self_attention = MultiHeadedAttention(num_heads, head_size)
        self.feed_forward_network = FeedForward(n_embd)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.layer_norm_2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # Adding x -> The purpose of the residual connection is to ensure that important
        # information from the input sequence is preserved and propagated through the network.
        # Also for improved gradient flow (no vanishing or exploding gradients)
        # Original paper => Self Attention -> Add and Layer_Norm -> Feed_Forward
        # More recently =>  Layer_Norm -> Self_Attention -> Add and Feed_Forward
        x = x + self.self_attention(self.layer_norm_1(x)) 
        x = x + self.feed_forward_network(self.layer_norm_2(x))
        return x
        
        
class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        # https://towardsdatascience.com/master-positional-encoding-part-i-63c05d90a0c3
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # Positional encoding
        self.blocks = nn.Sequential(*[Block(n_embd, num_heads) for _ in range(n_layers)])
        self.final_layer_norm = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size) # lanuage modelling head
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean = 0.0, std = 0.02)
            
    
    def forward(self, idx, targets = None):
        B, T = idx.shape
        
        # Both idx and targets have shape (B, T)
        
        # Forward pass through the whole decoder architecture
        tok_emb = self.token_embedding_table(idx) # (B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device = device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C) after broadcast and add
        x = self.blocks(x) # (B, T, C)
        x = self.final_layer_norm(x) # (B, T, C)
        logits = self.lm_head(x)  # (B, T, vocab_size)
        
        # Loss calculation - Cross Entropy (negative log likelihood loss)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # latter block_size part of the sequence
            idx_cond = idx[:, -block_size:]
            # compute the logits and loss
            logits, loss = self(idx_cond)
            # Pick from only the current timestep
            logits = logits[:, -1, :] # (B, T, C) to (B, C)
            probabilites = F.softmax(logits, dim = -1) # (B, C)

            # Sample from the distribution using the probabilites
            idx_next = torch.multinomial(probabilites, num_samples = 1)  # (B, 1)
            idx = torch.cat((idx, idx_next), dim = 1) # Append the sampled token(s) to the running sequence
        
        return idx
            
        
        
    