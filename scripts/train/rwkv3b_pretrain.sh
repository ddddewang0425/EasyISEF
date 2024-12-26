export CUDA_VISIBLE_DEVICES=0,1,2,3

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"
#CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 
# export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1  # InfiniBand 사용하지 않을 경우 비활성화
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_DEBUG_SUBSYS=ALL
export TORCH_DISTRIBUTED_DEBUG=INFO
export TORCH_SHOW_CPP_STACKTRACES=1
export TORCH_CUDA_ARCH_LIST=7.5
python train.py \
    --proj_dir out/rwkv3b-v060_pretrain \
    --data_file /home/gpuadmin/Desktop/RWKV/blip_laion_cc_sbu_558k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 18 --epoch_begin 0 --epoch_save 0 \
    --micro_bsz 1 --accumulate_grad_batches 1 --n_layer 32 --n_embd 2560 --pre_ffn 0 \
    --lr_init 1e-3 --lr_final 1e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 4 --precision fp16 --strategy deepspeed --grad_cp 1 \
    --image_folder /home/gpuadmin/Desktop/RWKV/images \
    --vision_tower_name /home/gpuadmin/Desktop/RWKV/myclip \
    --freeze_rwkv 0 --detail low --grid_size -1 --image_position no \
    --enable_progress_bar True