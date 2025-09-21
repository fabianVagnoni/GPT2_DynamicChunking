from dataloader import DataLoaderLite
import tiktoken
import torch
# from generate import generate
from gpt2_module import GPT, GPTConfig, set_device
from lr_scheduler import get_lr
import time
import os

# Distributed Data -------------------------------------------------------------------------------------------------

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.distributed as dist

# Set up DDP (Distributed Data Parallel)
# torch run sets env-vars RANK, LOCAL_RANK, WORLD_SIZE
ddp = int(os.environ.get("RANK", -1)) != -1 # Asks if this is a DDP run
print("DDP",ddp)
if ddp:
    assert torch.cuda.is_available(), "Cuda is needed for DDP"
    init_process_group(backend="nccl")
    ddp_rank=int(os.environ.get("RANK")) # Which GPU instance?
    ddp_world_size=int(os.environ.get("WORLD_SIZE")) # How many GPUs?
    ddp_local_rank=int(os.environ.get("LOCAL_RANK")) # Node number
    device, device_str = set_device()
    device=f"cuda:{ddp_local_rank}"
    master_process = ddp_rank == 0 # Keep printing just to process 0
else:
    # no ddp
    ddp_rank=0
    ddp_world_size=1
    ddp_local_rank=0
    device, device_str = set_device()
    master_process = True

# Hypeparams -------------------------------------------------------------------------------------------------

MAX_TEXT = 1000000
MODEL = "gpt2"
STEPS = 50
LR = 3e-4
WEIGHT_DECAY = 0.1

# Batch Sizes
DESIRED_B = 2*256 #524288 # 2**19, nice number near 500k
B = 16 # micro-batch size
T = 16
assert DESIRED_B % (B*T*ddp_world_size) == 0, "Make sure the desired batch size is divisible by the micro-batch (B) size times the sequence length (T)"
grad_accum_steps = DESIRED_B // (B*T*ddp_world_size) # We will foward-backward grad_accum_steps times W/O calling update, just += to the grad
if master_process:
    print(f"Desired Batch Size: {DESIRED_B} | Micro-Batch Size: {B} | Sequence Length: {T} | World Size: {ddp_world_size} => Gradient Accumulation Update: {grad_accum_steps} steps")

# -------------------------------------------------------------------------------------------------

def test_run():
    loader = DataLoaderLite(B,T,MODEL,ddp_rank,ddp_world_size,device)
    model = GPT(GPTConfig(vocab_size=50304)) # 50304 is a nice number, can be even divided by 128!!!
    model.to(device)
    torch.compile(model)
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
    raw_model = model.module if ddp else model # Always contains raw, unwrapped model
    optimizer = raw_model.configure_optimizer(weight_decay=WEIGHT_DECAY, lr=LR, device=device_str)

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
            # Disallow gradient sync until we have accumulated all grads
            if ddp:
                model.require_backward_grad_sync = (grad_step == grad_accum_steps-1) # Allow it in last
            loss.backward() # Implicit += to grads

        if ddp:
            dist.all_reduce(loss_accum, op=dist.ReduceOp.AVG) # Mean accross the different loss accums
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
        tokens_per_sec = grad_accum_steps * loader.B * loader.T * ddp_world_size / dt
        if master_process:
            print(f"Step {step} -> Loss: {loss_accum} | lr: {lr:.4e} | norm: {norm:.5f} | dt: {dt:.2f}s | tokens/sec: {tokens_per_sec:.2f}")

    # Close parallel processes
    if ddp:
        destroy_process_group()
    import sys; sys.exit(0)

test_run()
