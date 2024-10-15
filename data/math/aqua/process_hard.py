import random
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import argparse
import os
import torch
import numpy as np

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Load the dataset from Hugging Face
# dataset = load_dataset('deepmind/aqua_rat', split='train')

def load_model(args):
    toker = AutoTokenizer.from_pretrained(
    args.pretrained_model_path, use_fast=False,
    add_bos_token=False, add_eos_token=False,
    cache_dir='/mnt/nvme/guoqing/',
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.pretrained_model_path,
        # device_map='auto',
        cache_dir='/mnt/nvme/guoqing/',
        # attn_implementation="eager",
        use_safetensors=True,
        torch_dtype=torch.float16,
        # torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    ).to(device)
    model.eval()
    return model, toker


def generate_response(model, tokenizer, prompt, device, max_new_tokens=128):
    messages = [
    {"role": "system", "content": f"{sys_prompt}"},
    {"role": "user", "content": f"{prompt}"},
    ]
    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
        ).to(model.device)
    outputs=model.generate(
                input_ids,
                max_new_tokens=512,
                do_sample=False,
                eos_token_id=terminators,
                temperature=0.0,
                pad_token_id=toker.eos_token_id,
    )
    response=outputs[0][input_ids.shape[-1]:]
    response=toker.decode(response,skip_special_tokens=True)

    return response

def change_format(text):
    id=text.split(')')[0].strip()
    content=text.split(')')[1].strip()

    return f"{id}. {content}"


# Function to parse each sample and generate the required dictionary
def parse_sample(sample, index):
    rationale = sample['rationale']

    # perform cot reasoning
    question = sample['question']
    prompt = f"{question} Let's solve this step by step, using Step 1, Step 2, Step 3 and more if necessary."
    response = generate_response(model, toker, prompt, device, max_new_tokens=args.max_new_tokens)
    
    # Build the dictionary for this sample
    parsed_dict = {
        'index': index,
        'question': sample['question'],
        'answer': sample['answer'],
        'cot': response,
        'rationale': rationale,
        'counter options': sample['counter options']
    }
    
    return parsed_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument("--data_path", type=str, default='templates.json')
    parser.add_argument("--example_path", type=str, default='data/examples.txt')
    parser.add_argument("--template", type=str, default='The native language of [X] is [Y] .')
    parser.add_argument("--max_turns", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--max_num", type=int, default=10)
    parser.add_argument("--mode", type=str, default='max')

    args = parser.parse_args()

    set_seed(0)

    with open('data/math/aqua-rat/easy/counter-aqua.json') as f:
        dataset = json.load(f)

    print('loading model...')
    model, toker = load_model(args)

    sys_prompt = "You are a reliable assistant, consistently providing accurate information and answering my questions with the corresponding option. Please maintain your responses when you are confident in their accuracy, and promptly correct them if found to be incorrect."
    terminators = [
        toker.eos_token_id,
        toker.convert_tokens_to_ids("<|eot_id|>")
    ]
    # Parse the dataset
    add_num =2000
    dataset = dataset[1000: 1000+add_num]
    parsed_samples = [parse_sample(sample,idx) for idx,sample in tqdm(enumerate(dataset))]

    output_file='data/math/aqua-rat/hard/counter-aqua_1k-3k.json'
    json.dump(parsed_samples, open(output_file, 'w'), indent=2)

    # Example: Print the first parsed sample to see the structure
    print(parsed_samples[0])
