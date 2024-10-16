import datasets
import json
# from matplotlib import pyplot as plt
from tqdm import tqdm
import torch
import numpy as np
import random
import os

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # set_deterministic_environment(seed)

# Load the dataset
def get_lama():
    dataset = datasets.load_dataset("janck/bigscience-lama")

    template_list=[]
    template_dict={}
    out_file=open('templates.json','w', encoding='utf-8')
    for data in dataset['test']:
        template = data['template']
        if template not in template_dict:
            template_dict[data['template']]=[]
        if data['template'] not in template_list:
            template_list.append(data['template'])
        data.pop('template')
        data.pop('predicate_id')
        data.pop('type')
        template_dict[template].append(data)

    json.dump(template_dict, out_file, indent=2)
    return template_list,template_dict

def is_country(answer):
    # Check if the answer matches any country name
    for country in countries.values():
        if answer.lower() == country['name'].lower():
            return True
    return False

def analysis(out_file, max_turns):
    f=open(out_file, 'r', encoding='utf-8')
    data = json.load(f)
    sim_scores = [entry[f'{max_turns}_turn_conversations']['sim_score'] for entry in data]
    bully_success = [entry[f'{max_turns}_turn_conversations']['bully_success'] for entry in data]
    conv_turns = [entry[f'{max_turns}_turn_conversations']['conversation turns'] for entry in data]
    avg_conv_turns = sum(conv_turns)/len(conv_turns)
    total_bully_success = sum(bully_success)
    print("Total bully success:", total_bully_success)
    print("AVG conversation turns:", avg_conv_turns)
    
    # Calculate the Spearman correlation coefficient
    spearman_coefficient, p_value = spearmanr(sim_scores, bully_success)
    # Display the result
    print(spearman_coefficient, p_value)

def plot_template_distribution(template_dict):
    # 获取模板名称和对应的数量
    templates = list(template_dict.keys())
    counts = [len(samples) for samples in template_dict.values()]