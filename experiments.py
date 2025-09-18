from gpt2_module import GPT, GPTConfig, set_device
from generate import generate
import torch

device = set_device()
num_return_seq = 5
max_len = 30


#model = GPT.from_pretrained("gpt2")
model = GPT(GPTConfig())
model_name = "gpt2"
input_str = "Hello, I'm a language model,"
model.eval()

dec = generate(model_name, model, num_return_seq, max_len, input_str)