import os
os.environ["RWKV_JIT_ON"] = "1"

import json
from PIL import Image
import pandas as pd
import numpy as np
import math
import argparse
import torch
from pathlib import Path
from tqdm import tqdm
from src.rwkv_tokenizer import TRIE_TOKENIZER
from src.dataset import DEFAULT_IMAGE_TOKEN, DEFAULT_STOP_TOKEN, STOP_TOKEN_INDEX
from src.dataset import process_image_tokens_in_conversations, preprocess
from src.utils import Conversation, gpt4v_crop, load_image_from_base64
from transformers import CLIPImageProcessor
import torchvision


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]

def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False


def load_questions(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix
    if suffix == ".jsonl":
        questions = [json.loads(q) for q in open(file_path)]
    elif suffix == ".json":
        questions = json.load(open(file_path))
    elif suffix == ".tsv":
        questions = pd.read_table(file_path).to_dict("records")
    else:
        raise ValueError("Unsupported file type: {}".format(suffix))
    return questions


def get_question_id(line):
    if "question_id" in line:
        return line["question_id"]
    elif "id" in line:
        return line["id"]
    elif "index" in line:
        return line["index"]
    else:
        raise ValueError("Cannot find question id in line: {}".format(line))


def get_options(line, options):
    parsed_options = []
    for option in options:
        option_value = line[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


def get_input_text_mmbench(line, lang='en'):
    all_options = ['A', 'B', 'C', 'D']
    options = get_options(line, all_options)
    question = line['question']
    hint = line['hint']
    if not is_none(hint):
        question = hint + '\n' + question
    for option_char, option in zip(all_options[:len(options)], options):
        question = question + '\n' + option_char + '. ' + option
    question = DEFAULT_IMAGE_TOKEN + '\n' + question
    if lang == 'cn':
        question = question + '\n' + "请直接回答选项字母。"
    else:
        question = question + '\n' + "Answer with the option's letter from the given choices directly."
    return question
    

def get_input_text(line, dataset_name):
    if dataset_name == "mmbench":
        return get_input_text_mmbench(line)
    elif dataset_name == "mmbench_cn":
        return get_input_text_mmbench(line, lang='cn')
    else:
        if "text" in line:
            return DEFAULT_IMAGE_TOKEN + '\n' + line["text"]
        elif "conversations" in line:
            return line["conversations"][0]["value"]
        else:
            raise ValueError("Cannot find input text in line: {}".format(line))

import matplotlib.pyplot as plt
totensor = torchvision.transforms.ToTensor()
def get_input_image_tensor(line, image_folder, image_processor, detail):
    if "image" in line:
        image_file = line["image"]
        if image_folder is not None:
            image = Image.open(image_folder / image_file)
            plt.imshow(image)
            plt.show()
        else: # image is base64 encoded
            image = load_image_from_base64(image_file)
        if args.detail == 'high':
            image = [image] + gpt4v_crop(image)
            image_tensor = image_processor(images=image, return_tensors='pt')['pixel_values']
        else:
            image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values']
    else:
        # image does not exist in the data, fill with zeros
        if detail == 'high':
            crop_size = image_processor.crop_size
            image_tensor = torch.zeros(7, 3, crop_size['height'], crop_size['width'])
        else:
            crop_size = image_processor.crop_size
            image_tensor = torch.zeros(1, 3, crop_size['height'], crop_size['width'])
    return image_tensor, totensor(image).to(args.device).unsqueeze(0) # add batch dimension

def eval_model(args):

    

    from src.model_state import RWKV_II
    model_path = Path(args.model_path)
    model_name = model_path.parent.name
    # Model
    model = RWKV_II(args)
    msg = model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
    print("msg of loading model: ", msg)
    model = model.bfloat16().to(args.device)
    tokenizer = TRIE_TOKENIZER("/home/gpuadmin/Desktop/RWKV/MK1/src/rwkv_vocab_v20230424.txt")
    image_processor = CLIPImageProcessor.from_pretrained(args.vision_tower_name)

    questions = load_questions(args.question_file)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    image_folder = Path(args.image_folder) if args.image_folder is not None else None

    out_file = open(output_file, "w")
    pbar = tqdm(total=len(questions))
    update_every = len(questions) // 100
    for i, line in enumerate(questions):
        idx = get_question_id(line)
        input_text = get_input_text(line, dataset_name=args.dataset_name)

        conv = Conversation(id=idx, roles=["human", "gpt"], conversations=[])
        conv.append_message(conv.roles[0], input_text)
        conv.append_message(conv.roles[1], "")

        conversations = process_image_tokens_in_conversations(conv.conversations, image_position=args.image_position)

        image_tensor, image = get_input_image_tensor(line, image_folder, image_processor, args.detail)
        image_tensor = image_tensor.unsqueeze(0).bfloat16().to(args.device)

        data_dict = preprocess(
            conversations,
            tokenizer,
            has_image=False,
            ctx_len=args.ctx_len,
            pad_token_id=0,
            do_pad_to_max_length=False)
        
        input_ids = data_dict['input_ids'].unsqueeze(0).to(args.device)
        cur_prompt = data_dict['input_text']

        

        with torch.inference_mode():
            output_ids, output_logits, output_probs = model.generate_without_last_image(
                input_ids,
                images=image_tensor[image_tensor!=DEFAULT_IMAGE_TOKEN],
                real_images=image,
                max_spots = args.max_spots,
                do_sample=False,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                stop_token_idx=STOP_TOKEN_INDEX)

        output = tokenizer.decode(output_ids).split(DEFAULT_STOP_TOKEN)[0].strip()
        print(output)
        # avg logit
        avg_logit = sum(output_logits) / len(output_logits)
        # geometric mean of probs
        avg_prob = np.prod(output_probs) ** (1.0 / len(output_probs))

        out_str = json.dumps({"question_id": idx,
                              "prompt": cur_prompt,
                              "text": output,
                              "avg_logit": str(round(avg_logit, 3)),
                              "avg_prob": str(round(avg_prob, 3)),
                              "model_id": model_name,
                              "metadata": {
                                  "image_file": line.get("image", None),
                              }}, ensure_ascii=False)
        out_file.write(out_str + "\n")
        # update progress bar
        if update_every > 0:
            if i % update_every == 0 and i != 0:
                pbar.update(update_every)
        else:
            pbar.update(1)
        out_file.flush()
    out_file.close()
    pbar.close()

if __name__ == "__main__":

    class Args:
        def __init__(self):
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
            self.grad_cp = 0
            self.freeze_rwkv = 1
            self.freeze_proj = 1
            self.precision = "bf16"
            self.model_path = "/home/gpuadmin/Desktop/RWKV/MK1/model/VisualRWKV_baseline_3b.pth"
            self.max_spots = 10
            
            # 비전 모델 설정
            self.vision_tower_name = "/home/gpuadmin/Desktop/RWKV/myclip"
            self.load_model = ""
            self.grid_size = 8
            self.detail = "low"

            self.random_seed = -1
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self.image_folder = "/home/gpuadmin/Desktop/RWKV/MK1/tmp_image"
            self.question_file = "/home/gpuadmin/Desktop/RWKV/MK1/tmp_query1.json"
            self.output_file = "/home/gpuadmin/Desktop/RWKV/MK1/output.jsonl"
            self.temperature = 0.2
            self.top_p = None
            self.max_new_tokens = 128
            self.num_chunks = 1
            self.chunk_idx = 0
            self.dataset_name = "default"
            self.image_position = "middle"
            
    args = Args()

    os.environ["RWKV_HEAD_SIZE_A"] = str(args.head_size_a)
    os.environ["RWKV_CTXLEN"] = str(args.ctx_len)
    if args.dim_att <= 0:
        args.dim_att = args.n_embd
    if args.dim_ffn <= 0:
        args.dim_ffn = int((args.n_embd * 3.5) // 32 * 32) # default = 3.5x emb size

    eval_model(args)