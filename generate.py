import tiktoken
import torch
import torch.nn.functional as F
SEED = 42

def generate(model_name, model_instance, num_seq, max_len, input_str):
    # Prefix tokens
    enc = tiktoken.get_encoding(model_name)
    tokens = enc.encode(input_str)
    tokens = torch.tensor(tokens, dtype=torch.long)
    tokens = tokens.unsqueeze(0).repeat(num_seq, 1)
    x = tokens # (5,8)

    torch.manual_seed(SEED)
    while x.size(1) < max_len:
        # Foward the model to get logits
        with torch.no_grad():
            logits = model_instance(x) # (B,T,vocab)
            # Take logits at the last position
            logits = logits[:,-1,:] # (B,vocab)
            # get proba
            probs = F.softmax(logits, dim=-1)
            # Do top-k sampling of 5 (HF default) => (5,50), (5,50)
            topk_proba, topk_ind = torch.topk(probs, 50, dim=-1)
            # Sample a token from top-k
            ix = torch.multinomial(topk_proba,1) # (B,1)
            # Gather corresponding indices
            xcol = torch.gather(topk_ind, -1, ix) # (B,1)
            # Append to seq
            x = torch.cat((x,xcol),dim=1)

    for i in range(num_seq):
        tokens = x[i, :max_len].tolist()
        decoded = enc.decode(tokens)
        print(">", decoded)

    return decoded
