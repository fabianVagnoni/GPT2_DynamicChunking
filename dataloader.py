import tiktoken
import torch
# from generate import generate
from gpt2_module import GPT, GPTConfig, set_device
import time

MAX_TEXT = 1000000
MODEL = "gpt2"
STEPS = 50
B = 4
T = 16
LR = 3e-4
device, device_str = set_device()

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
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR,betas=(0.9,0.95),eps=1e-8) # Explicit GPT-3 paper hyperparams

    for i in range(STEPS):
        t0 = time.time()
        x,y = loader.next_batch()
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        if torch.cuda.is_available(): # Use Mixed Precision ONLY w/ GPU
            with torch.autocast(device_type=device_str, dtype=torch.bfloat16):
                logits, loss = model(x,y)
        else:
            logits, loss = model(x,y)
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm(model.parameters(), 1.0) # Print thenorm to supervise training. We want stable norm, spike may signal underlying issue
        optimizer.step()
        if torch.cuda.is_available():
            torch.cuda.synchronize() # "Wait for the GPU and the CPU to be on the same step"
        t1 = time.time()
        dt = t1 - t0
        tokens_per_sec = loader.B * loader.T / dt
        print(f"Step {i} -> Loss: {loss} | norm {norm:.5f} | dt: {dt:.2f}s | tokens/sec: {tokens_per_sec:.2f}")

test_loader()
