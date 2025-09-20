import math

MAX_STEPS = 50
WARMUP_STEPS = 10 
MAX_LR = 6e-4 # GPT-3 inspired
MIN_LR = MAX_LR * 0.1 # GPT-3 inspired
assert MAX_STEPS >= WARMUP_STEPS, "Max steps must be larger or equal than warm-up"

def get_lr(it):
    # 1. Linear warmup -> Linear progression up to MAX_LR
    if it < WARMUP_STEPS:
        return MAX_LR * (it+1) / WARMUP_STEPS 
    # 2. Min LR if we are in max_steps for decay
    elif it > MAX_STEPS:
        return MIN_LR
    # 3. Cosine decay from Max_LR
    decay_ratio = (it - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS) # % of intermediate steps we've taken
    assert 0 <= decay_ratio <= 1 # Assert %
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # Starts at cos(0)=1 and goes to 0
    return MIN_LR + coeff * (MAX_LR - MIN_LR)