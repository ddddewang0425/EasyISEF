########################################################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
########################################################################################################

import os, math, gc, importlib
import torch
# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
from transformers import CLIPVisionModel, CLIPImageProcessor
from torch.autograd import Variable
if importlib.util.find_spec('deepspeed'):
    import deepspeed

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam
from .dataset import IGNORE_INDEX, IMAGE_TOKEN_INDEX, STOP_TOKEN_INDEX
from .rwkv_tokenizer import TRIE_TOKENIZER
def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

from torch.utils.cpp_extension import load

HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
wkv6state_cuda = load(name="wkv6state", sources=["/home/gpuadmin/Desktop/RWKV/MK1/cuda/wkv6state_op.cpp", f"/home/gpuadmin/Desktop/RWKV/MK1/cuda/wkv6state_cuda_v1a.cu"],
                verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
# wkv6state_cuda = load(name="wkv6state", sources=["/home/gpuadmin/Desktop/RWKV/MK1/cuda/wkv6state_op.cpp", f"/home/gpuadmin/Desktop/RWKV/MK1/cuda/wkv6state_cuda_v1a.cu"],
#                 verbose=True, extra_cuda_cflags=[f"-D_N_={HEAD_SIZE}", f"-D_T_={int(os.environ['RWKV_CTXLEN'])}"])
    
class WKV_6STATE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, B, T, C, H, r, k, v, w, u, s):
        with torch.no_grad():
            assert HEAD_SIZE == C // H
            ctx.B = B
            ctx.T = T
            ctx.C = C
            ctx.H = H
            assert r.is_contiguous()
            assert k.is_contiguous()
            assert v.is_contiguous()
            assert w.is_contiguous()
            assert u.is_contiguous()
            assert s.is_contiguous()
            ctx.save_for_backward(r, k, v, w, u, s)
            y = torch.empty((B, T, C), device=r.device, dtype=r.dtype, memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6state_cuda.forward(B, T, C, H, r, k, v, w, u, s, y)
            return y

    @staticmethod
    def backward(ctx, gy):
        with torch.no_grad():
            assert gy.dtype == torch.bfloat16
            B = ctx.B
            T = ctx.T
            C = ctx.C
            H = ctx.H
            assert gy.is_contiguous()
            r, k, v, w, u, s = ctx.saved_tensors
            gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=gy.dtype, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=gy.dtype, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=gy.dtype, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gw = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=gy.dtype, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=gy.dtype, memory_format=torch.contiguous_format).uniform_(-100, 100)
            gs = torch.empty((B, H, C//H, C//H), device=gy.device, requires_grad=False, dtype=gy.dtype, memory_format=torch.contiguous_format).uniform_(-100, 100)
            wkv6state_cuda.backward(B, T, C, H, r, k, v, w, u, s, gy, gr, gk, gv, gw, gu, gs)
            gu = gu.contiguous()
            gu = Variable(torch.sum(gu, 0).view(H, C//H))
            return (None, None, None, None, gr, gk, gv, gw, gu, gs)

def RUN_CUDA_RWKV6STATE(B, T, C, H, r, k, v, w, u, s):
    return WKV_6STATE.apply(B, T, C, H, r, k, v, w, u, s)

########################################################################################################

class RWKV_Tmix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_maa_x = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_w = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_v = nn.Parameter(1.0 - (torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_maa_g = nn.Parameter(1.0 - torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            D_MIX_LORA = 32 # generate TIME_MIX for w,k,v,r,g
            if args.n_embd >= 4096:
                D_MIX_LORA = 64
            self.time_maa_w1 = nn.Parameter(torch.zeros(args.n_embd, D_MIX_LORA*5))
            self.time_maa_w2 = nn.Parameter(torch.zeros(5, D_MIX_LORA, args.n_embd).uniform_(-0.01, 0.01))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att, )
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(1,1,args.dim_att))

            D_DECAY_LORA = 64
            if args.n_embd >= 4096:
                D_DECAY_LORA = 128
            self.time_decay_w1 = nn.Parameter(torch.zeros(args.n_embd, D_DECAY_LORA))
            self.time_decay_w2 = nn.Parameter(torch.zeros(D_DECAY_LORA, args.dim_att).uniform_(-0.01, 0.01))

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att, eps=(1e-5)*(args.head_size_divisor**2))

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) - x

        xxx = x + xx * self.time_maa_x
        xxx = torch.tanh(xxx @ self.time_maa_w1).view(B*T, 5, -1).transpose(0, 1)
        xxx = torch.bmm(xxx, self.time_maa_w2).view(5, B, T, -1)
        mw, mk, mv, mr, mg = xxx.unbind(dim=0)

        xw = x + xx * (self.time_maa_w + mw)
        xk = x + xx * (self.time_maa_k + mk)
        xv = x + xx * (self.time_maa_v + mv)
        xr = x + xx * (self.time_maa_r + mr)
        xg = x + xx * (self.time_maa_g + mg)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        ww = torch.tanh(xw @ self.time_decay_w1) @ self.time_decay_w2
        w = self.time_decay + ww

        return r, k, v, g, w

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x, state=None):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g, w = self.jit_func(x)
        if state is None:
            state = torch.zeros((B, H, C//H, C//H), device=r.device, dtype=r.dtype)    
        x = RUN_CUDA_RWKV6STATE(B, T, C, H, r, k, v, w, u=self.time_faaaa, s=state)

        return self.jit_func_2(x, g), state

########################################################################################################

class RWKV_CMix_x060(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_maa_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))
            self.time_maa_r = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0))

        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x) - x
        xk = x + xx * self.time_maa_k
        xr = x + xx * self.time_maa_r

        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv
    
########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x060(args, layer_id)
        self.ffn = RWKV_CMix_x060(args, layer_id)

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x, state=None):
        if self.layer_id == 0:
            # print(f"ln0 : {self.ln0.weight.dtype}")
            # print(f"x : {x.dtype}")
            x = self.ln0(x)
        xp, state = self.att(self.ln1(x), state)
        x = x + xp
        x = x + self.ffn(self.ln2(x))

        return x, state


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        self.blocks = nn.ModuleList([Block(args, i) for i in range(args.n_layer)])
        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)
        self.automatic_optimization = False
        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)

    def forward(self, x, state=None):
        args = self.args
        # B, T, D = x.size()
        # assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        if args.dropout > 0:
            x = self.drop0(x)

        if state is None:
            state = [None] * args.n_layer

        for i, block in enumerate(self.blocks):
            if args.grad_cp == 1:
                x, state[i] = deepspeed.checkpointing.checkpoint(block, x, state[i])
            else:
                x, state[i] = block(x, state[i])
        x = self.ln_out(x)
        x = self.head(x)
        return x, state

class RWKV_II(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.rwkv = RWKV(args)
        if len(args.load_model) > 0:
            self.load_rwkv_from_pretrained(args.load_model)
        self.tokenizer = TRIE_TOKENIZER("/home/gpuadmin/Desktop/RWKV/MK1/src/rwkv_vocab_v20230424.txt")
        self.firstInput = [torch.tensor(self.tokenizer.encode(x),dtype=torch.long) for x in ["To answer the question", "where is the region of interest in the image?"]]
        self.secondInput = [torch.tensor(self.tokenizer.encode(x),dtype=torch.long) for x in ["The region of interest in image is", "where is the next region of interest in the image?"]]
        self.thirdInput = [torch.tensor(self.tokenizer.encode(x),dtype=torch.long)for x in ["Based on the image", "and the region of interests, Answer the question:"]]
        self.vit = CLIPVisionModel.from_pretrained(args.vision_tower_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower_name)
        self.vit.requires_grad_(False)
        self.proj = nn.Linear(self.vit.config.hidden_size, args.n_embd, bias=False)
        self.emb_spot = nn.Linear(args.vocab_size, 2)

    def load_rwkv_from_pretrained(self, path):
        self.rwkv.load_state_dict(torch.load(path, map_location="cpu"))
        rank_zero_info(f"Loaded pretrained RWKV from {path}")

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False
    
    def freeze_rwkv(self, num_layers_to_freeze):
        # freeze all layers including embedding and lm head
        if num_layers_to_freeze == self.args.n_layer:
            self.rwkv.requires_grad_(False)
        # otherwise, freeze only the first num_layers_to_freeze layers
        for i, block in enumerate(self.rwkv.blocks):
            if i < num_layers_to_freeze:
                for p in block.parameters():
                    p.requires_grad_(False)
            else:
                for p in block.parameters():
                    p.requires_grad_(True)

    def freeze_emb(self):
        self.rwkv.emb.requires_grad_(False)

    def freeze_proj(self):
        self.proj.requires_grad_(False)

    def freeze_emb_spot(self):
        self.emb_spot.requires_grad_(False)

    def forward(self, samples):
        x, targets, image_features = self.preparing_embedding(samples)
        # bidirectional
        if (samples['input_ids']==IMAGE_TOKEN_INDEX).any():
            print("FUCK")
        logits = self.forward_without_last_image(
            input_ids = samples['input_ids'], 
            images = samples['images'],
            real_images = samples['real_images'],
            max_spots = self.args.max_spots,
            do_sample = False,
            temperature = 0.0,
            top_p = 0.0,
            max_new_tokens = self.args.max_new_tokens,
            stop_token_idx = STOP_TOKEN_INDEX,
            )
        return logits, targets

    def bidirectional_forward(self, x, x_emb=None, state=None):
        args = self.args
        if state is None:
            state = [None] * self.args.n_layer

        if args.dropout > 0:
            x = self.rwkv.drop0(x)

        for i, block in enumerate(self.rwkv.blocks):
            do_reverse = (i % 2 == 1)
            if do_reverse: # reverse
                x[:, self.img_start:self.img_end, :] = x[:, self.img_start:self.img_end, :].flip(1)
            
            if args.grad_cp == 1:
                x, state[i] = deepspeed.checkpointing.checkpoint(block, x, state[i])
            else:
                x, state[i] = block(x, state[i])
            
            if do_reverse: # reverse back
                x[:, self.img_start:self.img_end, :] = x[:, self.img_start:self.img_end, :].flip(1)

        x = self.rwkv.ln_out(x)

        x = self.rwkv.head(x)

        return x, state
    
    def encode_images(self, images):
        B, N, C, H, W = images.shape
        images = images.view(B*N, C, H, W)
        image_features = self.vit(images).last_hidden_state
        L, D = image_features.shape[1], image_features.shape[2]
        # rerange [B*N, L, D] -> [B, N, L, D]
        image_features = image_features.view(B, N, L, D)[:, 0, :, :]
        image_features = self.grid_pooling(image_features)
        return self.proj(image_features)
    
    def grid_pooling(self, image_features):
        cls_features = image_features[:, 0:1, :]
        image_features = image_features[:, 1:, :] #drop cls token
        if self.args.grid_size == -1: # no grid pooling
            return torch.cat((image_features, cls_features), dim=1)
        if self.args.grid_size == 0: # take cls token
            return cls_features
        if self.args.grid_size == 1: # global avg pooling
            return torch.cat((image_features.mean(dim=1, keepdim=True), cls_features), dim=1)
        B, L, D = image_features.shape
        H_or_W = int(L**0.5)
        image_features = image_features.view(B, H_or_W, H_or_W, D)
        grid_stride = H_or_W // self.args.grid_size
        image_features = F.avg_pool2d(image_features.permute(0, 3, 1, 2), 
                                      padding=0,
                                      kernel_size=grid_stride, 
                                      stride=grid_stride)
        image_features = image_features.permute(0, 2, 3, 1).view(B, -1, D)
        return torch.cat((image_features, cls_features), dim=1)

    def get_max_image_token_indice(self, samples):
        max_image_token_indice = 0
        for cur_input_ids in samples["input_ids"]:
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 1:
                image_token_indice = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0][0]
                max_image_token_indice = max(max_image_token_indice, image_token_indice)
        return max_image_token_indice
    
    def truncate_input(self, new_input_embeds, new_labels):
        # prioritize retaining the labels at the beginning
        # if there are no valid labels at the beginning, retain the labels from the end
        truncated_input_embeds = []
        truncated_labels = []
        for x, y in zip(new_input_embeds, new_labels):
            valid_labels = [i for i in y[:self.args.ctx_len] if i != IGNORE_INDEX]
            if valid_labels:
                truncated_input_embeds.append(x[:self.args.ctx_len])
                truncated_labels.append(y[:self.args.ctx_len])
            else:
                truncated_input_embeds.append(x[-self.args.ctx_len:])
                truncated_labels.append(y[-self.args.ctx_len:])
        return truncated_input_embeds, truncated_labels
   
    def preparing_embedding(self, samples, truncate=True):
        device, label_dtype = samples["labels"].device, samples["labels"].dtype
        emb_dtype = samples["images"].dtype
        ### prepare image features
        image_features  = self.encode_images(samples["images"]) # with cls token
        ### prepare input token
        new_input_embeds = []
        new_labels = []
        max_image_token_indice = self.get_max_image_token_indice(samples)
        self.img_start = max_image_token_indice
        self.img_end = max_image_token_indice + (image_features.shape[1] - 1) # exclude cls token
        for idx, cur_input_ids in enumerate(samples["input_ids"]):
            cur_labels = samples["labels"][idx]
            cur_new_input_ids = torch.zeros(max_image_token_indice, dtype=cur_input_ids.dtype, device=device)
            cur_new_labels = torch.full((max_image_token_indice,), IGNORE_INDEX, device=device, dtype=label_dtype)
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0: # no image in this sample
                # mask image feature, set to 0
                image_features[idx] = torch.zeros_like(image_features[idx])
            elif num_images == 1: # only one image in this sample
                image_token_indice = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0][0]
                # first text part, left paded
                cur_new_input_ids[-image_token_indice:] = cur_input_ids[:image_token_indice]
                cur_new_labels[-image_token_indice:] = cur_labels[:image_token_indice]
            else:
                raise ValueError(f"Too many images in one sample: {num_images}, should be 0 or 1.")
            # convert to list
            cur_new_input_embeds = [self.rwkv.emb(cur_new_input_ids)]
            cur_new_labels = [cur_new_labels]
            # image part
            cur_image_features = image_features[idx]
            cur_new_input_embeds.append(cur_image_features)
            cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=device, dtype=label_dtype))
            # last text part
            if num_images == 1:
                cur_new_input_embeds.append(self.rwkv.emb(cur_input_ids[image_token_indice+1:]))
                cur_new_labels.append(cur_labels[image_token_indice+1:])
            else: # no image
                cur_new_input_embeds.append(self.rwkv.emb(cur_input_ids))
                cur_new_labels.append(cur_labels)
            # concat them
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)
        # Truncate sequences to max length as image embeddings can make the sequence longer
        # keep the first `ctx_len` tokens, to make sure instruction complete
        if truncate:
            new_input_embeds, new_labels = self.truncate_input(new_input_embeds, new_labels)
        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)
        new_input_embeds_padded = torch.zeros((batch_size, max_len, self.args.n_embd), dtype=emb_dtype, device=device)
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=label_dtype, device=device)
        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            new_input_embeds_padded[i, :cur_len] = cur_new_embed
            new_labels_padded[i, :cur_len] = cur_new_labels
        return new_input_embeds_padded, new_labels_padded, image_features
    
    def extract_spot(self, images, loc, spot_size=336):
        """
        image: [B, C, H, W] 형태의 원본 이미지
        loc: [B, 2] 형태의 상대 위치 (x, y 각각 0~1 사이 값)
        spot_size: 추출할 spot의 크기
        """
        spots = torch.zeros(len(images), images[0].shape[0], spot_size, spot_size, device=images[0].device)
        for b, image in enumerate(images):
            H, W = image.shape[-2:]
            
            # 상대 위치를 실제 픽셀 위치로 변환
            x = (loc[b, 0] * W).long()  # [B]
            y = (loc[b, 1] * H).long()  # [B]
            
            # spot의 시작점과 끝점 계산
            half_size = spot_size // 2
            start_x = x - half_size
            start_y = y - half_size
            end_x = x + half_size
            end_y = y + half_size
            
            
            # 원본 이미지에서 실제로 가져올 영역 계산
            valid_start_x = max(0, start_x)
            valid_start_y = max(0, start_y)
            valid_end_x = min(W, end_x)
            valid_end_y = min(H, end_y)
            
            # spot에서의 대응되는 위치 계산
            spot_start_x = max(0, -start_x)
            spot_start_y = max(0, -start_y)
            spot_end_x = spot_size - max(0, end_x - W)
            spot_end_y = spot_size - max(0, end_y - H)
            
            # 유효한 영역 복사
            spots[b, :, spot_start_y:spot_end_y, spot_start_x:spot_end_x] = \
                image[:, valid_start_y:valid_end_y, valid_start_x:valid_end_x]
            
        return spots

    """
    def generate(self, input_ids, images, real_images, max_spots, num_spots, do_sample, temperature, top_p, max_new_tokens, stop_token_idx) -> list[int]:
        ''' one mode to generate, only generate one sample at a time
        # input_ids: [1, seq_len]
        # images: [1, 1, 3, 224, 224]
        # do_sample: bool
        # temperature: float
        # top_p: float
        # max_new_tokens: int
        max_spots : A number of maximum spots to generate
        num_spots : A number of recent spots to use
        '''

        self.reset_state()
        FirstInput = torch.cat((torch.tensor([IMAGE_TOKEN_INDEX]).repeat(input_ids.shape[0], 1).to(input_ids.device), self.firstInput[0].repeat(input_ids.shape[0], 1), input_ids[input_ids != IMAGE_TOKEN_INDEX], self.firstInput[1].repeat(input_ids.shape[0], 1)),dim=0)        
        # prepare samples
        sampels = {"input_ids": FirstInput, "images": images, "labels": torch.full_like(FirstInput, IGNORE_INDEX)}
        # prepare embedding, x: [1, seq_len, n_embd]
        x, = self.preparing_embedding(sampels, truncate=False)
        x_total = x.clone().detach()
        # generate
        generated_tokens = []
        generated_token_logits = []
        generated_token_probs = []
        generated_spot_sizes = [x.shape[-2]]
        sum_generated_spot_size = x.shape[-2]
        for i in range(max_spots):
            logits = self.bidirectional_forward(x)[:, -1, :]
            if torch.argmax(logits, dim=-1, keepdim=True).item() == IMAGE_TOKEN_INDEX:
                break
            loc = torch.sigmoid(self.emb_spot(logits))
            spot = self.extract_spot(real_images, loc)
            spot_tensor = self.image_processor.preprocess(spot, return_tensors='pt')['pixel_values']
            SecondInput = torch.cat((self.secondInput[0].repeat(input_ids.shape[0], 1), torch.tensor([IMAGE_TOKEN_INDEX]).repeat(input_ids.shape[0], 1), self.secondInput[1].repeat(input_ids.shape[0], 1)),dim=0)            
            spot_sample = {"input_ids": SecondInput, "images": spot_tensor, "labels": torch.full_like(SecondInput, IGNORE_INDEX)}
            gx, = self.preparing_embedding(spot_sample, truncate=False)
            x = torch.cat((x, gx.clone().detach()), dim=-2)
            x_total = torch.cat((x_total, gx), dim=-2)
            generated_spot_sizes.append(gx.shape[-2])
            sum_generated_spot_size += gx.shape[-2]
            if len(generated_spot_sizes) > num_spots:
                x = x[:, generated_spot_sizes[0]:]
                sum_generated_spot_size -= generated_spot_sizes[0]
                generated_spot_sizes.pop(0)

        x = x[:, -self.args.ctx_len:, :]
        for i in range(max_new_tokens):
            if i==0:
                logits = self.bidirectional_forward(x_total)[:, -1, :]
            else:
                logits = self.bidirectional_forward(x)[:, -1, :]
            print(logits.shape)
            if do_sample:
                raise NotImplementedError
            else: # greedy
                # [1, vocab_size] -> [1, 1]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                next_token_logit = logits.gather(-1, next_token)
                probs = torch.softmax(logits, dim=-1)
                next_token_prob = probs.gather(-1, next_token)
            generated_tokens.append(next_token.item())
            generated_token_logits.append(next_token_logit.item())
            generated_token_probs.append(next_token_prob.item())
            if generated_tokens[-1] == stop_token_idx:
                break
            x = torch.cat((x, self.rwkv.emb(next_token)), dim=-2)
            x = x[:, -self.args.ctx_len:, :] # truncate
        return generated_tokens, generated_token_logits, generated_token_probs
    """

    def generate(self, input_ids, images, real_images, max_spots, do_sample, temperature, top_p, max_new_tokens, stop_token_idx) -> list[int]:
        ''' one mode to generate, only generate one sample at a time
        # input_ids: [1, seq_len]
        # images: [1, 1, 3, 224, 224]
        # do_sample: bool
        # temperature: float
        # top_p: float
        # max_new_tokens: int
        max_spots : A number of maximum spots to generate
        num_spots : A number of recent spots to use
        '''

                
        # prepare samples
        
        # prepare embedding, x: [1, seq_len, n_embd]
        
        # generate
        generated_tokens = []
        generated_token_logits = []
        generated_token_probs = []
        logits = None
        for i in range(max_spots):
            if i==0:
                FirstInput = torch.cat((torch.tensor([IMAGE_TOKEN_INDEX]).repeat(input_ids.shape[0], 1).to(input_ids.device), self.firstInput[0].to(input_ids.device).repeat(input_ids.shape[0], 1), input_ids, self.firstInput[1].to(input_ids.device).repeat(input_ids.shape[0], 1)),dim=1)
                sampels = {"input_ids": FirstInput, "images": images, "labels": torch.full_like(FirstInput, IGNORE_INDEX)}
                x,_,_ = self.preparing_embedding(sampels, truncate=False)
            else:
                loc = torch.sigmoid(self.emb_spot(logits))
                spot = self.extract_spot(real_images, loc)
                spot_tensor = self.image_processor.preprocess(spot, return_tensors='pt')['pixel_values'].to(input_ids.device).unsqueeze(1)
                SecondInput = torch.cat((self.secondInput[0].to(input_ids.device).repeat(input_ids.shape[0], 1), torch.tensor([IMAGE_TOKEN_INDEX]).to(input_ids.device).repeat(input_ids.shape[0], 1), self.secondInput[1].to(input_ids.device).repeat(input_ids.shape[0], 1)),dim=1)            
                spot_sample = {"input_ids": SecondInput, "images": spot_tensor, "labels": torch.full_like(SecondInput, IGNORE_INDEX)}
                x,_,_ = self.preparing_embedding(spot_sample, truncate=False)
            logits = self.bidirectional_forward(x)[:, -1, :]
            # if torch.argmax(logits, dim=-1, keepdim=True).item() == IMAGE_TOKEN_INDEX:
            #    break
        loc = torch.sigmoid(self.emb_spot(logits))
        spot = self.extract_spot(real_images, loc)
        spot_tensor = self.image_processor.preprocess(spot, return_tensors='pt')['pixel_values'].to(input_ids.device).unsqueeze(1)
        Second_Third_Input = torch.cat((self.secondInput[0].to(input_ids.device).repeat(input_ids.shape[0], 1), torch.tensor([IMAGE_TOKEN_INDEX]).to(input_ids.device).repeat(input_ids.shape[0], 1)), dim=1)
        Second_Third_sample = {"input_ids": Second_Third_Input, "images": spot_tensor, "labels": torch.full_like(Second_Third_Input, IGNORE_INDEX)}
        x,_,_ = self.preparing_embedding(Second_Third_sample, truncate=False)
        logits = self.bidirectional_forward(x)[:, -1, :]
        
        Third_Input = torch.cat((self.thirdInput[0].to(input_ids.device).repeat(input_ids.shape[0], 1), torch.tensor([IMAGE_TOKEN_INDEX]).to(input_ids.device).repeat(input_ids.shape[0], 1), self.thirdInput[1].to(input_ids.device).repeat(input_ids.shape[0], 1), input_ids), dim=1)
        Third_sample = {"input_ids": Third_Input, "images": images, "labels": torch.full_like(Third_Input, IGNORE_INDEX)}
        x,_,_ = self.preparing_embedding(Third_sample, truncate=False)
        for i in range(max_new_tokens):
            logits = self.bidirectional_forward(x)[:, -1, :]
            if do_sample:
                raise NotImplementedError
            else: # greedy
                # [1, vocab_size] -> [1, 1]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                next_token_logit = logits.gather(-1, next_token)
                probs = torch.softmax(logits, dim=-1)
                next_token_prob = probs.gather(-1, next_token)
            generated_tokens.append(next_token.item())
            generated_token_logits.append(next_token_logit.item())
            generated_token_probs.append(next_token_prob.item())
            if generated_tokens[-1] == stop_token_idx:
                break
            x = torch.cat((x, self.rwkv.emb(next_token)), dim=-2)
            x = x[:, -self.args.ctx_len:, :] # truncate
        return generated_tokens, generated_token_logits, generated_token_probs
    
    def forward_without_last_image(self, input_ids, images, real_images, max_spots, do_sample, temperature, top_p, max_new_tokens, stop_token_idx) -> list[int]:
        ''' one mode to generate, only generate one sample at a time
        # input_ids: [1, seq_len]
        # images: [1, 1, 3, 224, 224]
        # do_sample: bool
        # temperature: float
        # top_p: float
        # max_new_tokens: int
        max_spots : A number of maximum spots to generate
        num_spots : A number of recent spots to use
        '''
        # print(f"input_ids dtype: {input_ids.dtype} / images dtype: {images.dtype} / real_images dtype: {real_images[0].dtype}")
        generated_tokens = []
        generated_token_logits = []
        generated_token_probs = []
        logits = None
        for i in range(max_spots):
            if i==0:
                FirstInput = torch.cat((torch.tensor([IMAGE_TOKEN_INDEX]).repeat(input_ids.shape[0], 1).to(input_ids.device), self.firstInput[0].to(input_ids.device).repeat(input_ids.shape[0], 1), input_ids, self.firstInput[1].to(input_ids.device).repeat(input_ids.shape[0], 1)),dim=1)
                sampels = {"input_ids": FirstInput, "images": images, "labels": torch.full_like(FirstInput, IGNORE_INDEX)}
                x,_,_ = self.preparing_embedding(sampels, truncate=False)
            else:
                loc = torch.sigmoid(self.emb_spot(logits))
                spot = self.extract_spot(real_images, loc)
                spot_tensor = self.image_processor.preprocess(spot, return_tensors='pt', do_rescale=False)['pixel_values'].to(images.dtype).to(input_ids.device).unsqueeze(1)
                # print(f"spot_tensor : {spot_tensor.dtype}")
                SecondInput = torch.cat((self.secondInput[0].to(input_ids.device).repeat(input_ids.shape[0], 1), torch.tensor([IMAGE_TOKEN_INDEX]).to(input_ids.device).repeat(input_ids.shape[0], 1), self.secondInput[1].to(input_ids.device).repeat(input_ids.shape[0], 1)),dim=1)            
                spot_sample = {"input_ids": SecondInput, "images": spot_tensor, "labels": torch.full_like(SecondInput, IGNORE_INDEX)}
                x,_,_ = self.preparing_embedding(spot_sample, truncate=False)
            logits, state = self.bidirectional_forward(x)
            logits = logits[:,-1,:]
            # if torch.argmax(logits, dim=-1, keepdim=True).item() == IMAGE_TOKEN_INDEX:
            #    break
        loc = torch.sigmoid(self.emb_spot(logits))
        spot = self.extract_spot(real_images, loc)
        spot_tensor = self.image_processor.preprocess(spot, return_tensors='pt', do_rescale=False)['pixel_values'].to(images.dtype).to(input_ids.device).unsqueeze(1) 
        Second_Third_Input = torch.cat((self.secondInput[0].to(input_ids.device).repeat(input_ids.shape[0], 1), torch.tensor([IMAGE_TOKEN_INDEX]).to(input_ids.device).repeat(input_ids.shape[0], 1)), dim=1)
        Second_Third_sample = {"input_ids": Second_Third_Input, "images": spot_tensor, "labels": torch.full_like(Second_Third_Input, IGNORE_INDEX)}
        x,_,_ = self.preparing_embedding(Second_Third_sample, truncate=False)
        logits, state = self.bidirectional_forward(x, state=state)
        logits = logits[:,-1,:]

        Third_Input = torch.cat((self.thirdInput[0].to(input_ids.device).repeat(input_ids.shape[0], 1), torch.tensor(self.tokenizer.encode("images")).to(input_ids.device).repeat(input_ids.shape[0], 1), self.thirdInput[1].to(input_ids.device).repeat(input_ids.shape[0], 1), input_ids), dim=1)
        Third_sample = {"input_ids": Third_Input, "images": images, "labels": torch.full_like(Third_Input, IGNORE_INDEX)}
        x,_,_ = self.preparing_embedding(Third_sample, truncate=False)
        logits, _ = self.bidirectional_forward(x, state=state) # [B,T,Vocab]
        logits = logits[:,-input_ids.shape[1]:,:]
        tokens = torch.argmax(logits, dim=-1, keepdim=True)
        tokens_logits = logits.gather(-1,tokens).squeeze(-1)
        return logits

if __name__ == "__main__":
    B, T, C, H = 2, 4, 8, 2
    r = torch.randn((B, T, C), dtype=torch.bfloat16, device='cuda')
    k = torch.randn((B, T, C), dtype=torch.bfloat16, device='cuda')
    v = torch.randn((B, T, C), dtype=torch.bfloat16, device='cuda')
    w = torch.randn((B, T, C), dtype=torch.bfloat16, device='cuda')
    u = torch.randn((B, C), dtype=torch.bfloat16, device='cuda')
    s = torch.zeros((B, H, C // H, C // H), dtype=torch.bfloat16, device='cuda')

    try:
        y = RUN_CUDA_RWKV6STATE(B, T, C, H, r, k, v, w, u, s)
        print("Forward pass successful")
        y.sum().backward()
        print("Backward pass successful")
    except Exception as e:
        print(f"Error: {e}")