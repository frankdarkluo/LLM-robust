import random
from datasets import load_dataset
from tqdm import tqdm
import json

# Load the dataset from Hugging Face
with open('data/logic/reclor/reclor_data/train.json', 'r') as file:
    dataset = json.load(file)

random.seed(0)

# Function to parse each sample and generate the required dictionary
def parse_sample(sample, index):
    option_ids = ['A', 'B', 'C', 'D']
    # Extract the context, question, options, and correct answer index
    context = sample['context']
    question = sample['question']
    full_question = f"Context:\n{context}\nQuestion:{question}"
    
    options = sample['answers']
    correct_answer_index = sample['label']
    correct_answer_id = option_ids[correct_answer_index]
    correct_answer = f"{correct_answer_id}. {options[sample['label']]}"

    # Format the question to include options in the text
    options_text = "\n".join([f"{option_ids[i]}. {option}" for i, option in enumerate(options)])  # A, B, C, etc.
    full_question_text = f"{full_question}\nOptions:\n{options_text}\nAnswer: "
    
    # Get the wrong answers by excluding the correct one
    wrong_answers = [f'{option_ids[i]}. {opt}' for i, opt in enumerate(options) if i != correct_answer_index]
    
    # Randomly select two wrong answers as 'bully options'
    bully_options = random.sample(wrong_answers, 1)
    # bully_ids = [option_ids[options.index(opt)] for opt in bully_options]
    # bully_answers = [opt for opt in bully_options]
    
    # Build the dictionary for this sample
    parsed_dict = {
        'index': index,
        'question': full_question_text,
        'answer': correct_answer,
        'counter options': bully_options,
    }
    
    return parsed_dict

# Parse the dataset
parsed_samples = [parse_sample(sample,idx) for idx,sample in tqdm(enumerate(dataset))]

# Save to a JSON file
path='data/logic/reclor'
output_file = f'{path}/counter-reclor.json'
with open(output_file, 'w') as f:
    json.dump(parsed_samples, f, indent=2)

# Example: Print the first parsed sample to see the structure
print(parsed_samples[0])
