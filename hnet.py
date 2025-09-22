from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class HNetConfig:
    block_size: int = 1024
    vocab_size: int = 257
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 512
    dc_target_N: int = 3
    dc_alpha: float = 0.03
    pad_id: int = 256
    use_bias: bool = True

# -------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: HNetConfig):
        super().__init__()
        # Debug
        assert config.n_embd & config.n_head == 0
        # Key, Query, Value projections in Batch
        self.c_attn = nn.Linear(config.n_embd , 3 * config.n_embd)
        # Output Projections for residual
        self.c_proj = nn.Linear(config.n_embd , config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 # Attribute to signal that proj is a residual connection and must be scaled to (1/sqrt(n))
        # Reg
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Buffer for Bias/Mask => For masking future tokens
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                            .view(1, 1, config.block_size, config.block_size))

    def forward(self, x): # This is MHA batched operation
        B, T, C = x.size() # Batch size, Sequence Length, Embedding

        # Calculate Query, Key, Values for all heads in batch and 
        # Separate the batch (Possible because C = n_heads * head_size)
        qkv = self.c_attn(x) # (B,T,C) @ (C , C*3) => (B,T,C*3)
        q, k, v = qkv.split(self.n_embd, dim=2) # (B,T,C*3) => 3 * (B,T,C) 
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1,2) # (B, n_heads, T, head_size)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1,2) 
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1,2) 

        # FlashAttention -> We perform all operations on GPU memory w/o moving the big att matrix to HBM
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Projection
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Reassemble outputs by concatenating n_head w/ head_size => (B, T, C) [The contiguous "saved into a memory chunk" the view]
        y = self.c_proj(y)
        return y

# -------------------------------------------------------------------------------------------------

class MLP(nn.Module): # Two linnear proj sandwiched between a Gelu
    def __init__(self, config: HNetConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd) # Residual
        self.c_proj.NANOGPT_SCALE_INIT = 1.0 # Attribute to signal that proj is a residual connection and must be scaled to (1/sqrt(n))
        self.gelu = nn.GELU(approximate="tanh")
    
    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

# -------------------------------------------------------------------------------------------------

# Transformer Block with Layer Norms, MLP head & Attention Mechanism
class Block(nn.Module):
    def __init__(self, config: HNetConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config) # Map(?)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config) # Reduce(?)

    def forward(self, x):
        # Difference w.r.t. AttentionIsAllYouNeed:
        #    1. LayerNorms before attn
        #    2. LayerNorms are not included in Residual for cleaner gradient flow
        x = x + self.attn(self.ln_1(x)) 
        x = x + self.mlp(self.ln_2(x))
        return x

# -------------------------------------------------------------------------------------------------

class DynChunking(nn.Module):
    def __init__(self, n_embd, threshold, config: HNetConfig):
        super().__init__()
        self.Wkq = nn.Linear(n_embd,n_embd*2,bias=False)        
        self.threshold = threshold
        self.config = config

    def route(self, x, attn_mask=None):
        kq = self.Wkq(x)
        k,q = kq.split(self.config.n_embd, dim=2) # (B,T,C*2) => 2 * (B,T,C)
        k_prev = torch.roll(k, shifts=1, dims=1) # k{t-1}
        p = 0.5 * (1 - self._cos_sim(q, k_prev)) # (B,T)
        p[:,0] = 1.0 # Boundary start

        if attn_mask is not None: # Mask padding
            p = p.masked_fill(~attn_mask.bool(), 0.0)
        
        bt = (p >= self.threshold).to(x.dtype)
        bt[:, 0] = 1.0

        return p,bt

    def downsample(self, x, p, bt):
        B,T,C = x.shape

        # Check what indices to keep
        keep_ix = bt > 0.5 # (B,T), bool
        counts = keep_ix.sum(dim=1) # (B,) => How many items to keep per row, max will be padding
        Tds = int(counts.max().item()) # Padding index
        
        # Bringing keep_ix==True to the front to discard all other ix
        perm = torch.argsort(keep_ix.to(torch.int8), dim=1, descending=True, stable=True)  # (B, T), with index we want to keep in front
        sel = perm[:, :Tds]                       # (B, Tds), padded
        
        # Dechunk mask for position & gather chunks
        mask_ds = torch.arange(Tds, device=x.device).unsqueeze(0) < counts.unsqueeze(1) # (B, Tds), masking range above count
        x_chunks = torch.take_along_dim(x, sel.unsqueeze(-1).expand(-1, -1, C), dim=1) # (B, Tds, E)
        P_chunks = torch.take_along_dim(p, sel, dim=1) # (B, Tds)
        
        # Zero-Out padding
        x_chunks = x_chunks.masked_fill(~mask_ds.unsqueeze(-1), 0)
        P_chunks = P_chunks.masked_fill(~mask_ds, 0)
        gather_idx = sel.masked_fill(~mask_ds, 0)     
        
        # State for dechunking
        state = {
            "p_full": p, # (B, T)
            "b_full": bt, # (B, T)
            "gather_idx": gather_idx, # (B, Tds)
            "mask_ds": mask_ds, # (B, Tds)
        }

        return x_chunks, P_chunks, mask_ds, state

    @staticmethod
    def _cos_sim(a, b, eps=1e-8):
        a_n = a / (a.norm(dim=-1, keepdim=True) + eps)
        b_n = b / (b.norm(dim=-1, keepdim=True) + eps)
        return (a_n * b_n).sum(dim=-1) 

    @staticmethod
    def ratio_loss(p, b, mask, N=3, att_mask=None):
        B,T = p.shape
        att_mask = att_mask if att_mask else torch.ones((B,T))
        L = att_mask.sum(dim=1).clamp_min(1).float() # (B,)
        F = ((b > 0.5).float() * att_mask.float()).sum(dim=1) / L # ""
        G = (p * att_mask.float()).sum(dim=1) / L   # ""
        ratio = (N/(N-1)) * (((N - 1.0) * F * G) + ((1.0 - F) * (1.0 - G))) # ""

# -------------------------------------------------------------------------------------------------

class DeChunking(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def upsample(self, zt, state):
        B,T,C = zt.shape
        pt, bt, gather_idx, mask_ds = state["p_full"], state["b_full"], state["gather_idx"], state["mask_ds"]
        # Confidence scoring (Eq 6)
        ct = (pt ** bt) * (pt.new_zeros(pt.shape) - pt) ** bt # (B,T)
        # Gradient stabilization (Eq 7)
        ct_ste = ct + (1.0 - ct).detach() # (B,T)
        # Causal expansion (Eq 8)
        bt_int = (bt > 0.5).long() # (B,T)
        idx = torch.cumsum(bt_int, dim=1) - 1 # (B,T), determine what index to expand
        idx = idx.clamp(min=0, max=Tds - 1) # (B,T), cut to padding length
        i_arange = torch.arange(B, device=x_chunks.device).unsqueeze(1).expand(B, T) # (B, T)
        z_up = x_chunks[i_arange, idx]  # (B, Tds, C) => (B, T, C) 
        # Confidence-weighted decompression (9)
        z_up = ct.unsqueeze(-1) * z_up # (B,T,1)(Broadcasted into (B,T,C)) * (B,T,C) => (B,T,C)

    @staticmethod
    def ema(self, z, pt, eps=1e-12): # Eq (5)
        B, L, D = z.shape
        decay = (1.0 - P).clamp_min(eps) # [B, T]
        S = torch.cumsum(torch.log(decay), dim=1)  #  [B, T]

        # Build all pairwise differences S_t (B, T, 1) - S_k (B, T, T) => (B, T, T)
        delta = S.unsqueeze(2) - S.unsqueeze(1)

        # exp(Delta) gives prod_{j=k+1..t} (1-p_j); mask to lower triangle (k <= t)
        W = torch.tril(torch.exp(delta)) # (B, T, T)

        # Multiply each column k by p_k
        W = W * P.unsqueeze(1) # (B, T, T), broadcasted over k-dim

        # batched causal matmul: bar_z[b,t,:] = sum_k W[b,t,k] * z[b,k,:]
        bar_z = W @ z # (B, T, C)
        return bar_z


# -------------------------------------------------------------------------------------------------

class HNet(nn.Module):
    def __init__(self, config: HNetConfig):
        super().__init__()
        self.config = config
        self.embedding = HNetEmbedding(config.vocab_size, config.block_size, config.n_embd)

        self.encoder = nn.ModuleDict(dict(# Dictionary of modules, saves the keys
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.decoder = nn.ModuleDict(dict(# Dictionary of modules, saves the keys
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        self.m = nn.ModuleDict(dict(# Dictionary of modules, saves the keys
            h = nn.ModuleList([Block(config) for _ in range(int(config.n_layer*2))]),
            ln_f = nn.LayerNorm(int(config.n_embd*2))
        ))

        self.chunking = DynChunking(config.n_embd)
        self.dechunk = DeChunk()
