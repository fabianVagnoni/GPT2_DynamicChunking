import tiktoken
import torch
# from generate import generate
from gpt2_module import GPT, GPTConfig, set_device

MAX_TEXT = 1000000
MODEL = "gpt2"
STEPS = 50
B = 4
T = 32
LR = 3e-4
device = set_device()

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


def test_loader():
    loader = DataLoaderLite(B,T,MODEL,device)
    model = GPT(GPTConfig())
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(),lr=LR)

    for i in range(STEPS):
        x,y = loader.next_batch()
        x,y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits, loss = model(x,y)
        loss.backward()
        optimizer.step()
        print(f"Step {i} -> Loss: {loss}")
test_loader()
