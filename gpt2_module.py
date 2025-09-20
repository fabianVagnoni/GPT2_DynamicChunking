import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from transformers import GPT2LMHeadModel
import tiktoken
import inspect
SEED = 42

# -------------------------------------------------------------------------------------------------

# dataclass decorator automatically registers an __init__ method
# that saves all the attributes to self
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

# -------------------------------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
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

        # Attention
        # att = (q @ k.transpose(-2, -1) * (1.0 / T**(.5))) # (B, n_heads, T, head_size) @ (B, n_heads, head_size , T) = (B, n_heads, T, T)
        # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float("-inf")) # Where the mask is 0 (Upper Triangle of Future Tokens), fill -inf
        # att = F.softmax(att, dim=-1)
        # y = att @ v # (B, n_heads, T, T) @ (B, n_heads, T, head_size) => (B, n_heads, T, head_size)
        
        # FlashAttention -> We perform all operations on GPU memory w/o moving the big att matrix to HBM
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # Projection
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Reassemble outputs by concatenating n_head w/ head_size => (B, T, C) [The contiguous "saved into a memory chunk" the view]
        y = self.c_proj(y)
        return y

# -------------------------------------------------------------------------------------------------

class MLP(nn.Module): # Two linnear proj sandwiched between a Gelu
    def __init__(self, config: GPTConfig):
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
    def __init__(self, config: GPTConfig):
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

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Recreation of HuggingFace's layers of GPT2
        self.transformer = nn.ModuleDict(dict(# Dictionary of modules, saves the keys
            wte = nn.Embedding(config.vocab_size , config.n_embd), # Token Embeds 
            wpe = nn.Embedding(config.block_size , config.n_embd), # Post Embd
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f = nn.LayerNorm(config.n_embd)
        ))

        # FC head
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing scheme for WTE & the output head
        self.transformer.wte.weight = self.lm_head.weight # Now both weights share pointer in memory
        # We had 50,000 vocab_size and 768 hidden_dim => Each one was around 40M parameters
        # 40M out of 124M is around 30% ==> We're saving 30% of the model's parameters

        # Init weights following GPT-2
        self.apply(self._init_weights) # Apply() is a method of the nn.Module which applies a function iteratively to all submodules of your module
    
    def _init_weights(self, module, std=0.02):
        if isinstance(module, nn.Linear):
            if hasattr(module,"NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5 # 2 times number of layers because the transformer has two blocks that add to the residual pathway => MLP & Attn
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # Normal dist for linear layers w/ std 0.02
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias) # Zeros for biases
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=std) # Normal dist for emb layers w/ std 0.02

    def forward(self, idx, targets=None):
        loss = None
        # ids => (B, T) [Batch sequences of T ids]
        B, T = idx.size()
        assert T <= self.config.block_size, f"Cannot foward sequence of legth {T}. Block size is {self.config.block_size}"
        # Foward token and posit embd
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # Ints from 0 to T, (T)
        pos_embd = self.transformer.wpe(pos) # Posit Look-Up => (T, n_embd)
        tok_embd = self.transformer.wte(idx) # Token Look-Up => (T, n_embd)
        x = pos_embd + tok_embd
        # Foward the blocks
        for block in self.transformer.h:
            x = block(x)
        # foward the final layernorm and classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            # Logits (B,T,vocab_size) =(Flatten)=> (B*T,vocab_size) [-1 automatically reshapes]
            # targets (B,T) =(Flatten)=> (B*T)
        return logits, loss

    # Decorator that makes the method get as input the class
    # Used mainly when you have a method that should return the class itself
    @classmethod
    def from_pretrained(cls, model_type):
        "Loads model weights from pretrained"
        assert model_type in {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"} # Size by params
        from transformers import GPT2LMHeadModel
        print(f"Loading weights from pretrained GPT: {model_type}")

        # Defining architecture according to size
        config_args = { # Params
            "gpt2": dict(n_layer=12, n_head=12, n_embd=768), # 124M
            "gpt2-medium": dict(n_layer=24, n_head=16, n_embd=1024), # 350M
            "gpt2-large": dict(n_layer=36, n_head=20, n_embd=1280), # 774M
            "gpt2-xl": dict(n_layer=48, n_head=25, n_embd=1600) # 1558M
        }[model_type]
        config_args["vocab_size"] = 50257
        config_args["block_size"] = 1024

        # Initialize minGPT from scratch
        config = GPTConfig(**config_args)
        model = GPT(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith(".attn.bias")] # Discard bias bc it's an untrained buffer

        # Init a HF.Transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        # Copy while ensuring alignment of parameters' shapes and names
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.masked_bias")] # Discard bias bc it's an untrained buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith(".attn.bias")] # Discard bias bc it's an untrained buffer
        transposed = ["attn.c_attn.weight", "attn.c_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight"] # Weights that need to be transporsed bc of TensorFlow-Torch incompatibility
        assert len(sd_keys_hf) == len(sd_keys), f"Mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            # print(f"Loading weights: {k}")
            # print(f"HF Shape: {sd_hf[k].shape}")
            # print(f"SD Shape: {sd[k].shape}")
            # print("-"*100)
            if any(k.endswith(w) for w in transposed):
                # Conv1D needs transposing
                assert sd_hf[k].shape[::-1] == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k].t())
            else:
                # Simple copy
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])

        return model

    def configure_optimizer(self, weight_decay, lr, device):
        # Start w/ all candidate params (that require gra)
        param_dict = {pn : p for pn, p in self.named_parameters()}
        param_dict = {pn : p for pn, p in param_dict.items() if p.requires_grad}

        # Create optim Groups -> 2D or more dim params will be decayed | 1D won't
        # v.g., all biases and layer norm won't
        decay_params = [p for n,p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for n,p in param_dict.items() if p.dim() < 2]
        opt_groups = [ # Decay is useful to "bring down" params sizes such that one param doesn't drive too much info. It makes not much sense to include it in 1D
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0}
        ]
        num_decay = sum(p.numel() for p in decay_params)
        no_num_decay = sum(p.numel() for p in no_decay_params)
        print(f"Num Decay Param tensors {len(decay_params)}, with {num_decay:,} params")
        print(f"Num Decay Param tensors {len(no_decay_params)}, with {no_num_decay:,} params")
        # Create optimizer using fused for efficiency, if available
        fused_aval = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_aval and "cuda" in device
        print(f"Fused AdamW: {use_fused}")
        optimizer = torch.optim.AdamW(opt_groups, lr=lr, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
        return optimizer

# -------------------------------------------------------------------------------------------------

def set_device():
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    if device.type != "cuda" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    print(f"Using device: {device}")

    # Setting Seed & Precision TF16
    torch.manual_seed(1337)
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.cuda.manual_seed(1337)
    return device, device_str