import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.main = nn.Sequential(
            nn.Linear(config.embedding_dim, 4*config.embedding_dim), # blows up dim
            nn.GELU(),
            nn.Linear(4*config.embedding_dim, config.embedding_dim)
        )

    def forward(self, x):
        return self.main(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.qkv_proj = nn.Linear(config.embedding_dim, 3*config.embedding_dim)
        sl = config.seq_length
        #self.register_buffer('mask', torch.tril(torch.ones(sl, sl)).view(1, 1, sl, sl))

        self.num_heads = config.num_heads
        self.embedding_dim = config.embedding_dim

        self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim)

    def forward(self, x):
        b, t, c = x.size() # batch, seq length, size

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(self.embedding_dim, dim=2)
        q = q.view(b, t, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch, head dim, seq length (T), size
        k = k.view(b, t, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(b, t, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        #attn = q @ k.transpose(2, 3) / math.sqrt(k.size(-1)) # batch, head dim, T, T
        #attn = attn.masked_fill(self.mask[:,:,:t,:t]==0, float('-inf')) # trim mask to current sequence length
        #attn = F.softmax(attn, dim=-1)
        #out = attn @ v
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        
        out = out.permute(0, 2, 1, 3).reshape(b, t, c)
        return self.out_proj(out)
    

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