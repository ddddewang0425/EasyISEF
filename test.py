import deepspeed
a = load_checkpoint(f"/home/gpuadmin/Desktop/RWKV/checkpoints/best_val/bf16_zero_pp_rank_{deepspeed.comm.get_rank()}_mp_rank_00_optim_states.pt")
print(a)
