import tiktoken
import torch
# from generate import generate
from gpt2_module import GPT, GPTConfig, set_device
from lr_scheduler import get_lr
import time

MAX_TEXT = 1000000
MODEL = "gpt2"
STEPS = 50
LR = 3e-4
WEIGHT_DECAY = 0.1
device, device_str = set_device()

# Batch Sizes
DESIRED_B = 2*256 #524288 # 2**19, nice number near 500k
B = 16 # micro-batch size
T = 16
assert DESIRED_B % (B*T) == 0, "Make sure the desired batch size is divisible by the micro-batch (B) size times the sequence length (T)"
grad_accum_steps = DESIRED_B // (B*T) # We will foward-backward grad_accum_steps times W/O calling update, just += to the grad
print(f"Desired Batch Size: {DESIRED_B} | Micro-Batch Size: {B} | Sequence Length: {T} => Gradient Accumulation Update: {grad_accum_steps} steps")

# -------------------------------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, model_name, device):
        self.B = B
        self.T = T

        #load tokens from disk and store them in memory
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding(model_name)
        self.tokens = torch.tensor(enc.encode(text)) # No move to GPU now to save on memory
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"{len(self.tokens) // (B*T)} batches per epoch")

        # state
        self.current_position = 0
    
    def next_batch(self):
        B,T = self.B,self.T
        next_batch = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = next_batch[:-1].view(B,T) # inputs
        y = next_batch[1:].view(B,T) # recursive targets

        # advance position
        self.current_position += B*T

        # If loading next batch would be out of bounds, start from beggining
        if self.current_position + B*T + 1 > len(self.tokens):
            self.current_position = 0
        return x,y

# -------------------------------------------------------------------------------------------------

def test_loader():
    loader = DataLoaderLite(B,T,MODEL,device)
    model = GPT(GPTConfig(vocab_size=50304)) # 50304 is a nice number, can be even divided by 128!!!
    model.to(device)
    torch.compile(model)
    # optimizer = torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),eps=1e-8) # Explicit GPT-3 paper hyperparams
    optimizer = model.configure_optimizer(weight_decay=WEIGHT_DECAY, lr=LR, device=device_str)

    for step in range(STEPS):
        t0 = time.time()
        optimizer.zero_grad()
        loss_accum = 0.0
        for grad_step in range(grad_accum_steps):
            x,y = loader.next_batch()
            x,y = x.to(device), y.to(device)
            if torch.cuda.is_available(): # Use Mixed Precision ONLY w/ GPU
                with torch.autocast(device_type=device_str, dtype=torch.bfloat16):
                    logits, loss = model(x,y)
            else:
                logits, loss = model(x,y)
            loss = (1/grad_accum_steps) * loss # Normalized loss to avoid mere summation
            loss_accum += loss.detach()
            loss.backward() # Implicit += to grads

        norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0) # Print thenorm to supervise training. We want stable norm, spike may signal underlying issue
        # Determine LR given the scheduler
        lr = get_lr(step)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize() # "Wait for the GPU and the CPU to be on the same step"
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = grad_accum_steps * loader.B * loader.T / dt
        print(f"Step {step} -> Loss: {loss_accum} | lr: {lr:.4e} | norm: {norm:.5f} | dt: {dt:.2f}s | tokens/sec: {tokens_per_sec:.2f}")

test_loader()
