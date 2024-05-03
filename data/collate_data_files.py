# Take in a list of scored data files with {"prefix": "...", "continuation": "...", "score": "..."} in each row
# and output a single scored data file with {"prefix": "...", "continuation": "...", "score": "..."} in each row
# with a threshold for the score.

import jsonlines
from datetime import datetime

data_files = [  '../output/scored_toxic_benign_l3-meta-llama-Meta-Llama-3-8B-Instruct-214909.jsonl', 
                '../output/scored_toxic_benign_mistral-mistralai-Mistral-7B-Instruct-v0.2-223309.jsonl'
             ]

# positive_data_files = [  '../output/scored_toxic_benign_l3-meta-llama-Meta-Llama-3-8B-Instruct-214909.jsonl', 
#                 '../output/scored_toxic_benign_mistral-mistralai-Mistral-7B-Instruct-v0.2-223309.jsonl'
#              ]

# negative_data_files = [  '../output/scored_toxic_toxic_l3-meta-llama-Meta-Llama-3-8B-Instruct-233659.jsonl', 
#                 '../output/scored_toxic_toxic_l3-mistralai-Mistral-7B-Instruct-v0.2-002319.jsonl'
#              ]

type_of_data_files = 'ppo_train_data'
score_threshold = 0.35

def collate_files(data_files, type_of_data_files, score_threshold):

    result = []
    for file_path in data_files:
        with jsonlines.open(file_path, 'r') as reader:
            for line in reader:
                if float(line['score']) >= score_threshold:
                    result.append(line)

    print(f"Number of lines in the output file: {len(result)}")

    # Current time in YYYYMMDD format
    current_date = datetime.now().strftime('%Y%m%d')
    # Output path should be '../output/{type_of_data_files}_{current_date}.jsonl'
    output_file_path = f'../output/{type_of_data_files}_{current_date}.jsonl' 

    with jsonlines.open(output_file_path, 'w') as writer:
        for line in result:
            writer.write(line)

def collate_pairs_file(positive_data_files, negative_data_files, type_of_data_files, score_threshold):
    result_positive = []
    result_negative = []
    for file_path in positive_data_files:
        with jsonlines.open(file_path, 'r') as reader:
            for line in reader:
                if float(line['score']) >= score_threshold:
                    result_positive.append(line)

    for file_path in negative_data_files:
        with jsonlines.open(file_path, 'r') as reader:
            for line in reader:
                if float(line['score']) < score_threshold:
                    result_negative.append(line)

    final_result = []

    for positive_line in result_positive:
        for negative_line in result_negative:
            if positive_line['prefix'] == negative_line['prefix']:
                temp = {'prefix': positive_line['prefix'], 'positive_continuation': positive_line['continuation'], 'negative_continuation': negative_line['continuation'], 'positive_score': positive_line['score'], 'negative_score': negative_line['score']}
                final_result.append(temp)

    print(f"Number of lines in the output file: {len(final_result)}")

    # Current time in YYYYMMDD format
    current_date = datetime.now().strftime('%Y%m%d')
    # Output path should be '../output/{type_of_data_files}_{current_date}.jsonl'
    output_file_path = f'../output/mixed_{type_of_data_files}_{current_date}.jsonl' 

    with jsonlines.open(output_file_path, 'w') as writer:
        for line in final_result:
            writer.write(line)

# collate_pairs_file(positive_data_files, negative_data_files, type_of_data_files, score_threshold)
collate_files(data_files, type_of_data_files, score_threshold)