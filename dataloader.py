import torch
import tiktoken

# -------------------------------------------------------------------------------------------------

class DataLoaderLite:
    def __init__(self, B, T, model_name, process_rank, num_processes, device):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes

        #load tokens from disk and store them in memory
        with open("input.txt", "r") as f:
            text = f.read()
        enc = tiktoken.get_encoding(model_name)
        self.tokens = torch.tensor(enc.encode(text)) # No move to GPU now to save on memory
        print(f"Loaded {len(self.tokens)} tokens")
        print(f"{len(self.tokens) // (B*T)} batches per epoch")

        # state
        self.current_position = self.B * self.T * self.process_rank
    
    def next_batch(self):
        B,T = self.B,self.T
        next_batch = self.tokens[self.current_position : self.current_position + B*T + 1]
        x = next_batch[:-1].view(B,T) # inputs
        y = next_batch[1:].view(B,T) # recursive targets

        # advance position
        self.current_position += B*T*self.num_processes

        # If loading next batch would be out of bounds, start from beggining
        if self.current_position + B*T*self.num_processes + 1 > len(self.tokens):
            self.current_position = self.B * self.T * self.process_rank
        return x,y