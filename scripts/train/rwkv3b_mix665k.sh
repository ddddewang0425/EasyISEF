export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 切换到脚本所在目录的上两级目录
cd "$(dirname "$(dirname "$0")")/.."

# 打印当前工作目录
echo "Current working directory: $(pwd)"


python train.py --model_path out/rwkv3b-v060_pretrain/rwkv-1.pth \
    --wandb "" --proj_dir out/rwkv3b-v060_mix665k \
    --data_file /houhaowenT/huggingface_datasets/LLaVA-Instruct-150K/shuffled_llava_v1_5_mix665k.json \
    --data_type "json" --vocab_size 65536 \
    --ctx_len 2048 --epoch_steps 1000 --epoch_count 20 --epoch_begin 0 --epoch_save 5 \
    --micro_bsz 8 --accumulate_grad_batches 2 --n_layer 32 --n_embd 2560 --pre_ffn 0 \
    --lr_init 2e-5 --lr_final 2e-5 --warmup_steps 0 --beta1 0.9 --beta2 0.99 --adam_eps 1e-8 \
    --accelerator gpu --devices 8 --precision bf16 --strategy deepspeed_stage_1 --grad_cp 1 \
    --image_folder /houhaowenT/huggingface_datasets/LLaVA-Instruct-150K/images/ \
    --vision_tower_name /houhaowenT/huggingface_models/openai/clip-vit-large-patch14-336 \
    --freeze_rwkv 0 --freeze_proj 0 --detail low --grid_size -1 --image_position middle \
    --enable_progress_bar True
