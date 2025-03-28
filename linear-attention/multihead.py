from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import numpy as np
import math
from tqdm import tqdm
from modules import MultiHeadAttention, MLP

text = open('input.txt', 'r').read().strip().replace('3', 'three')
vocab = list(set(list(text)))
vocab.sort()

@dataclass
class GPTConfig:
    seq_length: int = 256
    vocab_size: int = len(vocab)
    num_heads: int = 8
    embedding_dim: int = 64
    num_blocks: int = 12
    batch_size: int = 128
    linear_attention: bool = False
    compile: bool = True

def sinusoidal_encoding(seq_len, dim, max_timescale=10000):
    PE = np.empty((dim, seq_len))
    pos = np.arange(seq_len).reshape(1, -1)
    i = np.arange(dim).reshape(-1, 1)
    inv = max_timescale ** (2/dim * i//2)
    PE[::2,:] = np.sin(pos / inv[::2])
    PE[1::2,:] = np.cos(pos / inv[1::2])
    return torch.tensor(PE)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.ln1 = nn.LayerNorm(config.embedding_dim)
        self.ln2 = nn.LayerNorm(config.embedding_dim)
        if config.linear_attention:
            raise NotImplementedError
        else:
            self.attention = MultiHeadAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        x = self.attention(self.ln1(x)) + x
        x = self.mlp(self.ln2(x)) + x
        return x
    
class GPT(nn.Module):
    def __init__(self, config, device):
        super().__init__()

        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.embedding_dim).to(device)
        self.pos_emb = sinusoidal_encoding(config.seq_length, config.embedding_dim).T.to(device)
        self.blocks = nn.ModuleList(TransformerBlock(config) for _ in range(config.num_blocks)).to(device)
        self.ln = nn.LayerNorm(config.embedding_dim).to(device)
        self.lm_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False).to(device)
        
        self.token_emb.weight = self.lm_head.weight # input and output embeddings use the same transformation

    def forward(self, tokens):
        b, t = tokens.size()
        assert t <= self.config.seq_length, "Cannot convert tokens longer than max sequence length"
        x = self.token_emb(tokens) # b, t -> b, t, c
        pos_emb = self.pos_emb[:t,:]
        x = (x + pos_emb).to(x.dtype)

        for block in self.blocks:
            x = block(x)

        x = self.ln(x)
        return self.lm_head(x) # logits

    def generate(self, input_tokens, seq_length=None, temperature=1.):
        tokens = input_tokens

        if seq_length is None:
            seq_length = self.config.seq_length

        while tokens.size(1) < seq_length:
            with torch.no_grad():
                  logits = self(tokens)[:, -1, :] # get last time dimension
                  logits = logits / temperature
                  probs = F.softmax(logits, dim=-1)
                  topk_probs, topk_idxs = torch.topk(probs, 20, dim=-1)

                  #idx_ix = torch.multinomial(topk_idxs.to(torch.float32), 1)
                  #idx = topk_idxs.index_select(dim=-1, index=idx_ix.flatten())
                  #tokens = torch.cat((tokens, torch.tensor([idx], device=tokens.device).to(tokens.dtype).view(1, 1)), dim=-1)
                  
                  topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
                  idx = torch.multinomial(topk_probs, num_samples=1)
                  token = topk_idxs.gather(dim=-1, index=idx)
                  tokens = torch.cat((tokens, token), dim=1)
        
        return tokens
    
class Tokenizer(nn.Module):
    def __init__(self, corpus):
        super().__init__()

        self.vocab = list(set(list(corpus)))
        self.vocab.sort()
        tokens = range(len(vocab))
        self.char_to_token = dict(zip(vocab, tokens))
        self.token_to_char = dict(zip(tokens, vocab))

    def forward(self, text):
        return [self.char_to_token[c] for c in list(text)]

    def decode(self, tokens):
        return ''.join(self.token_to_char[t] for t in tokens)
    

class DataLoader:
    def __init__(self, tokenizer, corpus, b, t):
        self.batch_size = b
        self.seq_length = t

        self.corpus = tokenizer(corpus)
        self.tokens = torch.tensor(self.corpus, dtype=torch.long)

        self.current_batch = 0

    def next_batch(self):
        tokens = self.tokens[self.current_batch:self.current_batch + self.batch_size*self.seq_length + 1]
        inputs = tokens[:-1].view(self.batch_size, self.seq_length)
        labels = tokens[1:].view(self.batch_size, self.seq_length)

        self.current_batch += self.batch_size*self.seq_length
        if self.current_batch + self.batch_size*self.seq_length + 1 > len(self.corpus):
            self.current_batch = 0

        return inputs, labels
    

def test_run():
    input = "To be or not to be, "
    input = tokenizer(input)
    tokens = torch.tensor(input, dtype=torch.long, device=device).unsqueeze(0)
    out = gpt.generate(tokens, seq_length=100).detach().cpu()
    out = tokenizer.decode(out.flatten().tolist())
    print(out)

def train():
    num_epochs = 100
    batches_per_epoch = 272 * 10 # 10 full loops

    scaler = GradScaler()
    for epoch in range(num_epochs):
        epoch_loss = 0
        print(f'\nEpoch {epoch+1} of {num_epochs}')
        for _ in tqdm(range(batches_per_epoch)):
            #torch.cuda.empty_cache()
            inputs, labels = dataloader.next_batch()
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()

            with autocast('cuda'):
                logits = gpt(inputs)
                #labels = F.one_hot(labels, num_classes=config.vocab_size).float()
                #loss = F.cross_entropy(logits, labels)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
                
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(gpt.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            torch.cuda.synchronize()
            epoch_loss += loss.item()

        torch.save(gpt.state_dict(), f'gpt_weights_{epoch+1}.pth')
        print(f'Loss: {epoch_loss/batches_per_epoch}')

        test_run()


if __name__ == "__main__":
    config = GPTConfig(linear_attention=False)
    tokenizer = Tokenizer(text)
    dataloader = DataLoader(tokenizer, text, b=config.batch_size, t=config.seq_length)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    gpt = GPT(config, device)
    if config.compile:
        gpt = torch.compile(gpt)
    optimizer = torch.optim.AdamW(gpt.parameters(), lr=0.0005, betas=(0.9, 0.95))
    train()
