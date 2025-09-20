Calc % Decrease from -ln(1/Options)

Consider:
  - How will we initialize the dyn chunk layers (Check paper for guidance)
  - 


Plan:
  - Modify the GPT class to include a custom tokenizer
  - Build H-Net Class
    ------------------------
    Loader
    1. Convert text to UTF-8
    ------------------------
    H-Net
    2. Embedd bytes into R(L x D) ==> nn.Embedding(vocab_size, D), where vocab_size is the number of bytes
    3. Encoder E(R(L x D)) -> 