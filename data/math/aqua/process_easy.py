import random
from datasets import load_dataset
from tqdm import tqdm

# Load the dataset from Hugging Face
dataset = load_dataset('deepmind/aqua_rat', split='train')

def change_format(text):
    id=text.split(')')[0].strip()
    content=text.split(')')[1].strip()

    return f"{id}. {content}"

random.seed(0)
# Function to parse each sample and generate the required dictionary
def parse_sample(sample, index):
    option_ids=['A','B','C','D','E']
    # Extract the question, options, correct answer index, and rationale
    question = sample['question']

    options = sample['options']
    correct_answer_index = ord(sample['correct']) - ord('A')  # Convert letter to index (A = 0, B = 1, etc.)
    correct_answer = options[correct_answer_index]
    # correct_id=correct_answer.split(')')[0].strip()
    # correct_answer= correct_answer.split(')')[1].strip()
    rationale = sample['rationale']
    
    # Format the question to include options in the text
    options_text = "\n".join([f"{option.split(')')[0].strip()}. {option.split(')')[1].strip()}" for i, option in enumerate(options)])  # A, B, C, etc.
    full_question_text = f"{question}\nOptions:\n{options_text}\nAnswer: "

    # Get the wrong answers by excluding the correct one
    wrong_answers = [opt for i, opt in enumerate(options) if i != correct_answer_index]
    
    # Randomly select two wrong answers as 'bully options'
    bully_options = [change_format(i) for i in random.sample(wrong_answers, 1)]
    # bully_ids = [opt.split(')')[0].strip() for opt in bully_options]
    # bully_answers = [opt.split(')')[1].strip() for opt in bully_options]
    
    # Build the dictionary for this sample
    parsed_dict = {
        'index': index,
        'question': full_question_text,
        # 'options': options,
        # 'correct id': correct_id,
        'answer': change_format(correct_answer),
        'rationale': rationale,
        'counter options': bully_options
    }
    
    return parsed_dict

# Parse the dataset
parsed_samples = [parse_sample(sample,idx) for idx,sample in tqdm(enumerate(dataset))]

import json
output_file='data/math/aqua-rat/easy/counter-aqua.json'
json.dump(parsed_samples, open(output_file, 'w'), indent=2)

# Example: Print the first parsed sample to see the structure
print(parsed_samples[0])
