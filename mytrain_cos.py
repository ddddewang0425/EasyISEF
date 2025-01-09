import torch
from datetime import timedelta
class Args:
    def __init__(self):
        self.name = "sum_spot3"
        self.proj_dir = f"/home/gpuadmin/Desktop/RWKV/out/{self.name}"
        self.train_data_file = "/home/gpuadmin/Desktop/RWKV/llavacosdataset/train_cos.json"
        self.valid_data_file = "/home/gpuadmin/Desktop/RWKV/llavacosdataset/valid_cos.json"
        self.data_type = "json"
        self.max_epochs = 1000
        self.train_epoch_steps = 20
        self.valid_epoch_steps = 10
        self.epoch_begin = 0
        self.epoch_save = 20
        self.micro_bsz = 8
        self.accumulate_grad_batches = 4
        self.accelerator = "gpu"
        self.devices = 4
        self.precision = "bf16"
        self.grad_cp = 1
        self.enable_progress_bar = True
        self.wandb = ""
        self.run_name = "demo_run"

        self.lr_init = 6e-4
        self.lr_final = 1e-5
        self.warmup_steps = -1
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.adam_eps = 1e-8
        self.weight_decay = 0.0
        self.weight_decay_final = -1
        self.ds_bucket_mb = 200
        self.print_param_shape = 0

        # 모델 기본 설정
        self.load_model = ""
        self.vocab_size = 65536
        self.ctx_len = 2048
        self.n_layer = 32
        self.n_embd = 2560
        self.dim_att = 0  # n_embd와 동일하게 설정
        self.dim_ffn = 0  # n_embd * 3.5 (rounded to multiple of 32)
        self.pre_ffn = 0
        self.head_size_a = 64
        self.head_size_divisor = 8
        self.dropout = 0
        self.freeze_rwkv = 0
        self.freeze_proj = 0
        self.precision = "bf16"
        self.dtype = torch.bfloat16
        self.model_path = "/home/gpuadmin/Desktop/RWKV/model/VisualRWKV_baseline_3b.pth"
        self.load_checkpoint_dir = f"/home/gpuadmin/Desktop/RWKV/checkpoints/spot3"
        self.loadormodel = "model"
        self.save_checkpoint_dir = f"/home/gpuadmin/Desktop/RWKV/checkpoints/{self.name}"
        self.max_spots = 3
        
        # 비전 모델 설정
        self.vision_tower_name = "/home/gpuadmin/Desktop/RWKV/myclip"
        self.load_model = ""
        self.grid_size = 1
        self.detail = "low"

        self.random_seed = 235

        self.image_folder = "/home/gpuadmin/Desktop/RWKV/llavacosdataset"
        self.temperature = None
        self.top_p = None
        self.max_new_tokens = 128
        self.num_chunks = 1
        self.chunk_idx = 0
        self.dataset_name = "default"
        self.image_position = "no"
        self.num_nodes = 1

        self.delta = 20
        self.answer_loss = 1.0
        self.spot_loss = 20.0
args = Args()

import os, warnings, math, datetime, sys, time, json
import numpy as np
from torch.utils.data import DataLoader, random_split
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.nn import functional as F
from tqdm import tqdm

# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"
os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# os.environ["TORCH_USE_CUDA_DSA"] = '1'
print("VISIBLE_DEVICES: ", os.environ["CUDA_VISIBLE_DEVICES"])
print("DEVICE_COUNT: ", torch.cuda.device_count())
print("CURRENT_DEVICE: ", torch.cuda.current_device())

with open('config.json', 'r') as f:
    deepspeed_config = json.load(f)
args.micro_bsz = deepspeed_config["train_micro_batch_size_per_gpu"]
args.accumulate_grad_batches = deepspeed_config["gradient_accumulation_steps"]
deepspeed_config["optimizer"]["params"]["lr"] = args.lr_init
deepspeed_config["optimizer"]["params"]["betas"] = (args.beta1, args.beta2)
deepspeed_config["optimizer"]["params"]["eps"] = args.adam_eps
deepspeed_config["optimizer"]["params"]["weight_decay"] = args.weight_decay
deepspeed_config["scheduler"]["params"]["warmup_min_lr"] = args.lr_init
deepspeed_config["scheduler"]["params"]["warmup_max_lr"] = args.lr_final
deepspeed_config["scheduler"]["params"]["warmup_num_steps"] = args.warmup_steps
# print(deepspeed_config)
np.set_printoptions(precision=4, suppress=True, linewidth=200)
warnings.filterwarnings("ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings("ignore", ".*The progress bar already tracks a metric with the*")
# os.environ["WDS_SHOW_SEED"] = "1"

args.my_timestamp = datetime.datetime.today().strftime("%Y-%m-%d-%H-%M-%S")
args.enable_checkpointing = False
args.replace_sampler_ddp = False
args.logger = False
args.gradient_clip_val = 1.0
args.num_sanity_val_steps = 0
args.check_val_every_n_epoch = int(1e20)
args.log_every_n_steps = int(1e20)
args.betas = (args.beta1, args.beta2)
args.real_bsz = int(args.num_nodes) * int(args.devices) * args.micro_bsz
os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
if args.dim_att <= 0:
    args.dim_att = args.n_embd
if args.dim_ffn <= 0:
    args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

#args.run_name = f"{args.vocab_size} ctx{args.ctx_len} L{args.n_layer} D{args.n_embd}"
if not os.path.exists(args.proj_dir):
    os.makedirs(args.proj_dir)

try:
    deepspeed_version = deepspeed.__version__
except:
    deepspeed_version = None
    pass

assert args.data_type in ["json"]
assert args.precision in ["fp32", "tf32", "fp16", "bf16"]
os.environ["RWKV_FLOAT_MODE"] = args.precision
os.environ["RWKV_JIT_ON"] = "1"
os.environ["NCCL_P2P_DISABLE"] = "1"

if deepspeed_config["zero_optimization"]["stage"] == 3:
    os.environ["RWKV_JIT_ON"] = "0"
    
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
if args.precision == "fp32":
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
else:
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True

from src.dataset_llavacos import MyDataset
from src.rwkv_tokenizer import TRIE_TOKENIZER
from transformers import AutoImageProcessor

args.tokenizer = TRIE_TOKENIZER("src/rwkv_vocab_v20230424.txt")
args.image_processor = AutoImageProcessor.from_pretrained(args.vision_tower_name)


from src.model_state_torch_spot import RWKV_II
# 256gb cpu memory is not enough for 8 gpus
# to use 6 gpus on 256gb cpu memory, use .half() to save memory
model = RWKV_II(args).to(args.dtype)

if args.loadormodel == "model":
    msg = model.load_state_dict(torch.load(args.model_path, map_location='cpu'), strict=False)
    model = model.to(args.dtype)
if args.freeze_rwkv > 0:
    model.freeze_rwkv(args.freeze_rwkv)
if args.freeze_proj > 0:
    model.freeze_proj()
model.freeze_emb() # freeze emb all the time

optimizer = DeepSpeedCPUAdam(
    model.parameters(),
    lr=args.lr_init,
    betas=args.betas,
    eps=args.adam_eps,
    weight_decay=args.weight_decay
)

torch.distributed.init_process_group(backend='nccl', timeout=timedelta(hours=24))
model_engine, optimizer, _, _ = deepspeed.initialize(
    model=model,
    optimizer=optimizer,
    config=deepspeed_config
)
if args.loadormodel == "load":
    load_successful, client_state = model_engine.load_checkpoint(args.load_checkpoint_dir, "best_val")
    assert load_successful, "best_val load failure"
deepspeed_config["zero_optimization"]["allgather_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
deepspeed_config["zero_optimization"]["reduce_bucket_size"] = args.ds_bucket_mb * 1000 * 1000
# print(deepspeed_config)

# print(model_engine.global_rank)
# must set shuffle=False, persistent_workers=False (because worker is in another thread)
device = model_engine.local_rank if hasattr(model_engine, 'local_rank') else torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(f"device : {device} / global_rank : {model_engine.global_rank}")
from src.dataset_llavacos import IGNORE_INDEX, IMAGE_TOKEN_INDEX, STOP_TOKEN_INDEX
import multiprocessing
args.global_rank = model_engine.global_rank
# dataset
if args.loadormodel=="load":
    my_epoch = multiprocessing.Value('i',client_state["epoch"])
    args.epoch_begin = client_state["epoch"]
else:
    my_epoch = multiprocessing.Value('i', 0)
    args.epoch_begin = 0
train_dataset = MyDataset(args, my_epoch, args.train_epoch_steps, cospath=args.train_data_file)
valid_dataset = MyDataset(args, my_epoch, args.valid_epoch_steps, cospath=args.valid_data_file)
args.vocab_size = train_dataset.vocab_size

def custom_collate_fn(batch):
    keys = batch[0].keys()
    values = [m.values() for m in batch]
    zipped_values = zip(*values)
    batch = {k:list(v) for k,v in zip(keys,zipped_values)}
    batch["input_ids"] = torch.stack(batch["input_ids"])
    batch["labels"] = torch.stack(batch["labels"]).to(args.dtype)
    batch["images"] = torch.stack(batch["images"]).to(args.dtype)
    batch["locs"] = torch.tensor(batch["locs"]).to(args.dtype)
    # for i,x in enumerate(batch["real_images"]):
    #     batch["real_images"][i]=x.to(args.dtype)
    del batch["input_text"]
    return batch

train_loader = DataLoader(
    train_dataset,
    shuffle=False,  # 섞지 않음
    pin_memory=True,
    batch_size=args.micro_bsz,
    num_workers=1,
    persistent_workers=False,
    drop_last=True,
    collate_fn = custom_collate_fn
)

val_loader = DataLoader(
    valid_dataset,
    shuffle=False,  # 섞지 않음
    pin_memory=True,
    batch_size=args.micro_bsz,
    num_workers=1,
    persistent_workers=False,
    drop_last=True,
    collate_fn = custom_collate_fn
)

def move_to_device(batch, device):
    for key, value in batch.items():
        if key!="real_images":
            batch[key] = value.to(device)
        # else:
        #     for i,ri in enumerate(value):
        #         batch[key][i] = value[i].to(device)
    return batch

import json

# 히스토리 및 best model 관리 변수
history = {
    "train_loss": [],
    "val_loss": [],
    "train_accuracy": [],
    "val_accuracy": [],
    "train_iou": [],
    "val_iou": []
}
best_val_loss = float('inf')
best_model_path = os.path.join(args.proj_dir, "best_model.pth")
history_path = os.path.join(args.proj_dir, "history.json")

def calculate_iou(pred_boxes, label_boxes):
    """
    Calculate IoU for batched ROI data.

    Args:
        pred_boxes (torch.Tensor): Predicted bounding boxes, shape (B, 3) where 3 = [x, y, w].
        label_boxes (torch.Tensor): Ground truth bounding boxes, shape (B, 3) where 3 = [x, y, w].

    Returns:
        torch.Tensor: IoU values for each box in the batch, shape (B,).
    """
    # Extract coordinates for predicted boxes
    pred_x1 = pred_boxes[:, 0]
    pred_y1 = pred_boxes[:, 1]
    pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2]
    pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 2]

    # Extract coordinates for label boxes
    label_x1 = label_boxes[:, 0]
    label_y1 = label_boxes[:, 1]
    label_x2 = label_boxes[:, 0] + label_boxes[:, 2]
    label_y2 = label_boxes[:, 1] + label_boxes[:, 2]

    # Calculate intersection coordinates
    inter_x1 = torch.max(pred_x1, label_x1)
    inter_y1 = torch.max(pred_y1, label_y1)
    inter_x2 = torch.min(pred_x2, label_x2)
    inter_y2 = torch.min(pred_y2, label_y2)

    # Calculate intersection area
    inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    # Calculate areas of predicted and label boxes
    pred_area = (pred_x2 - pred_x1) * (pred_y2 - pred_y1)
    label_area = (label_x2 - label_x1) * (label_y2 - label_y1)

    # Calculate union area
    union_area = pred_area + label_area - inter_area

    # Calculate IoU
    iou = inter_area / torch.clamp(union_area, min=1e-6)

    return iou.mean()

def calculate_accuracy(logits, labels):
    # logits: [batch, seq_len, vocab_size]
    # labels: [batch, seq_len]
    # IGNORE_INDEX를 제외한 위치에서 argmax(logit) == label 인 비율을 측정
    IGNORE_INDEX = -100  # 이미 상단에서 정의된 값과 맞출 것
    with torch.no_grad():
        preds = torch.argmax(logits, dim=-1)  # [batch, seq_len]
        valid_mask = (labels != IGNORE_INDEX)
        valid_labels = labels[valid_mask]
        valid_preds = preds[valid_mask]

        if valid_labels.numel() == 0:
            return 0.0
        correct = (valid_preds == valid_labels).sum().item()
        total = valid_labels.numel()
        accuracy = correct / total
        return accuracy

def validate(model_engine, val_loader, device):
    model_engine.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    total_iou = 0.0
    IGNORE_INDEX = -100
    start_time = 0
    end_time = 0
    with torch.no_grad():
        if deepspeed.comm.get_rank()==0:
            progressbar = tqdm(enumerate(val_loader), total=len(val_loader), desc=f"Valid {epoch+1}/{args.max_epochs}")
        else:
            progressbar = enumerate(val_loader)
        for step, batch in progressbar:
            batch = move_to_device(batch, device)
            logits, targets, locs = model_engine(batch)
            locs = locs.to(device)
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = targets[..., 1:].contiguous()

            # loss 계산
            valid_lengths = (shift_labels != IGNORE_INDEX).sum(1)
            valid_lengths = torch.max(valid_lengths, torch.ones_like(valid_lengths))
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32),
                shift_labels.view(-1).to(torch.long),
                ignore_index=IGNORE_INDEX,
                reduction='none'
            ).to(args.dtype)
            loc_loss = F.huber_loss(
                locs[0].to(torch.float32),
                batch["locs"].to(torch.float32),
                delta=args.delta,
                reduction='mean'
            ).to(args.dtype)

            loss = loss.view(shift_labels.size()).sum(1)/valid_lengths
            loss = loss.mean()

            # accuracy 계산
            accuracy = calculate_accuracy(shift_logits, shift_labels)
            iou = calculate_iou(locs[0].to(torch.float32), batch["locs"].to(torch.float32))

            total_loss += loss.item() + loc_loss.item()
            total_accuracy += accuracy
            total_iou += iou

    avg_loss = total_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    avg_accuracy = total_accuracy / len(val_loader)
    return avg_loss, avg_accuracy, avg_iou

# 메인 학습 루프
for epoch in range(args.epoch_begin, args.max_epochs):
    train_dataset.set_epoch(epoch)
    model_engine.train()

    train_loss = 0.0
    train_accuracy = 0.0
    train_iou = 0.0
    IGNORE_INDEX = -100
    time_logs = {"data_loading" : [], "inference" : [], "loss_accuracy" : [], "step" : []}
    if deepspeed.comm.get_rank() == 0:
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Train {epoch+1}/{args.max_epochs}")
    else:
        progress_bar = enumerate(train_loader)
    start_time = time.time()
    for step, batch in progress_bar:
        time_logs["data_loading"].append(time.time()-start_time);start_time=time.time()
        logits, targets, locs = model_engine(move_to_device(batch, device))
        time_logs["inference"].append(time.time()-start_time);start_time=time.time()

        locs = locs.to(device)
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        
        valid_lengths = (shift_labels != IGNORE_INDEX).sum(1)
        valid_lengths = torch.max(valid_lengths, torch.ones_like(valid_lengths))

        del logits, targets
        torch.cuda.empty_cache()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)).to(torch.float32),
            shift_labels.view(-1).to(torch.long),
            ignore_index=IGNORE_INDEX,
            reduction='none'
        ).to(args.dtype)
        loc_loss = F.huber_loss(
            input=locs[0].to(torch.float32),
            target=batch["locs"].to(torch.float32),
            delta=args.delta,
            reduction="mean"
        ).to(args.dtype)
        loss = loss.view(shift_labels.size()).sum(1)/valid_lengths
        loss = loss.mean()
        loss = loss*args.answer_loss + loc_loss*args.spot_loss
        # Accuracy 계산
        accuracy = calculate_accuracy(shift_logits, shift_labels)
        iou = calculate_iou(locs[0].to(torch.float32), batch["locs"].to(torch.float32))

        train_loss += loss.item()
        train_accuracy += accuracy
        train_iou += iou

        time_logs["loss_accuracy"].append(time.time()-start_time);start_time=time.time()

        model_engine.backward(loss)
        model_engine.step()

        time_logs["step"].append(time.time()-start_time);start_time=time.time()
    
    deepspeed.comm.barrier(device_ids=[0,1,2,3])
    if deepspeed.comm.get_rank()==0:print(", ".join([f"{k} : {sum(v)/len(v)}s" for k,v in time_logs.items()]))


    train_loss = train_loss / len(train_loader)
    train_accuracy = train_accuracy / len(train_loader)
    train_iou = train_iou / len(train_loader)

    # Validation
    val_loss, val_accuracy, val_iou = validate(model_engine, val_loader, device)
    deepspeed.comm.barrier(device_ids=[0,1,2,3])

     # History 저장
    if torch.distributed.is_initialized():
        # 모든 GPU에서 loss/accuracy 값을 모은 후, 평균화
        # 필요하다면 all_reduce를 통해 평균화할 수 있음.
        # 예:
        train_loss_tensor = torch.tensor(train_loss).to(device)
        val_loss_tensor = torch.tensor(val_loss).to(device)
        train_accuracy_tensor = torch.tensor(train_accuracy).to(device)
        val_accuracy_tensor = torch.tensor(val_accuracy).to(device)
        train_iou_tensor = torch.tensor(train_iou).to(device)
        val_iou_tensor = torch.tensor(val_iou).to(device)
        torch.distributed.all_reduce(train_loss_tensor)
        torch.distributed.all_reduce(val_loss_tensor)
        torch.distributed.all_reduce(train_accuracy_tensor)
        torch.distributed.all_reduce(val_accuracy_tensor)
        torch.distributed.all_reduce(train_iou_tensor)
        torch.distributed.all_reduce(val_iou_tensor)
        train_loss = train_loss_tensor.item() / torch.distributed.get_world_size()
        val_loss = val_loss_tensor.item() / torch.distributed.get_world_size()
        train_accuracy = train_accuracy_tensor.item() / torch.distributed.get_world_size()
        val_accuracy = val_accuracy_tensor.item() / torch.distributed.get_world_size()
        train_iou = train_iou_tensor.item() / torch.distributed.get_world_size()
        val_iou = val_iou_tensor.item() / torch.distributed.get_world_size()
        pass
    if deepspeed.comm.get_rank() == 0:
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        print(f"Epoch [{epoch+1}/{args.max_epochs}] completed.")
        print(f" Train Loss: {train_loss:.4f} | Train Accuracy: {train_accuracy:.4f} | Train Iou: {train_iou:.4f}")
        print(f" Valid Loss: {val_loss:.4f} | Valid Accuracy: {val_accuracy:.4f} | Valid Iou: {val_iou:.4f}")
    if args.epoch_save != 0 and epoch % args.epoch_save == 0 and val_loss >= best_val_loss:
        if deepspeed.comm.get_rank()==0:
            start_time=time.time()
            print("model auto saving... ",end='')
        # checkpoint = {'state_dict': model_engine.state_dict(),'optimizer' :optimizer.state_dict()}
        # torch.save(checkpoint, f'/home/gpuadmin/Desktop/RWKV/checkpoints/epoch_{epoch}.pth')
        model_engine.save_checkpoint(args.save_checkpoint_dir,f'epoch_{epoch}',client_state={"epoch":epoch})
        if deepspeed.comm.get_rank()==0:
            print(f"complete! in {time.time()-start_time}s")


        
    if val_loss < best_val_loss:
        if deepspeed.comm.get_rank() == 0:
            # best model 저장
            start_time=time.time()
            print("Best Model! Saving...",end='')
        # checkpoint = {'state_dict': model_engine.state_dict(),'optimizer' :optimizer.state_dict()}
        # torch.save(checkpoint, f'/home/gpuadmin/Desktop/RWKV/checkpoints/best_{epoch}_rank_{deepspeed.comm.get_rank()}.pth')
        model_engine.save_checkpoint(args.save_checkpoint_dir,f'best_val',client_state={"epoch":epoch})
        if deepspeed.comm.get_rank() == 0:
            # lean_state_dict = deepspeed.checkpoint.utils.clone_tensors_for_torch_save(model_engine.module.state_dict())
            print(f"complete! in {time.time()-start_time}s")
            print(f"Best model updated at epoch {epoch+1} with val_loss {val_loss:.4f} (decreases by {best_val_loss - val_loss})")
        best_val_loss = val_loss

    deepspeed.comm.barrier(device_ids=[0,1,2,3])



