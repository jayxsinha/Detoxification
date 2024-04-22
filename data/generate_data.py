#!/usr/bin/env python
# coding: utf-8
import os
os.environ['HF_HOME'] = '/project/pi_mccallum_umass_edu/jsinha_umass_edu'
import transformers
import torch
import time
import pandas as pd
import numpy as np
import jsonlines # useful
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoTokenizer
import argparse
from datetime import datetime
import pprint
import json
import warnings
warnings.filterwarnings('ignore')

def get_model_and_tokenizer(model_id):
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        load_in_8bit=True,
        trust_remote_code=True
    )
    # model.config.use_cache = False
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


class JigsawDataset(Dataset):
    def __init__(self, prefs):
        self.prefs=prefs
    def __getitem__(self, idx):
        return self.prefs[idx]
    def __len__(self):
        return len(self.prefs)

def collate_fn(batch, tokenizer, device = 'cuda'):
    tokenized_batch = tokenizer(batch, return_tensors='pt', padding='max_length', max_length=256).to(device)
    return tokenized_batch

def get_texts(input_prefs, prompt):
    output_prefs =[f"{prompt}\n\nSentence: \"{p}\"\nContinuation: " for p in input_prefs]
    return output_prefs

def get_dataloader(input_prefs, prompt, tokenizer, batch_size=4, shuffle=True, collate_fn=None, device='cuda'):
    input_texts = get_texts(input_prefs, prompt)
    input_dataset = JigsawDataset(input_texts)
    input_dataloader = DataLoader(input_dataset, shuffle=shuffle, batch_size=batch_size, collate_fn=lambda batch: collate_fn(batch, tokenizer))
    return input_dataloader

def dump_to_file(filename, prefs, continuations):
    with jsonlines.open(filename, mode='w') as writer:
        for (pref, cont) in zip(prefs, continuations):
            for cont in continuations:
                writer.write({"prefix": pref, "continuation": cont})


def generate_completions(args, model, tokenizer, input_dataloader, batch_size=4, test = False):
    completions = []
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(input_dataloader)):
            num = batch['input_ids'].shape[0]
            # print(batch['input_ids'].shape)
            model_output = model.generate(**batch,
                                         do_sample=True,
                                         top_p=args.top_p,
                                         temperature=args.temperature,
                                         max_new_tokens=args.max_new_tokens,
                                         num_return_sequences=args.num_returns_sequences,
                                         max_length=args.max_length)
            # print("model_output shape: ", model_output.shape)
            model_output = model_output[:, 256: ]
            generated_outputs = tokenizer.batch_decode(model_output, skip_special_tokens=True)
            generated_outputs = np.array(generated_outputs).reshape(num, -1).tolist()
            completions.extend(generated_outputs)
            if test:
                break
    return completions

def create_folder_if_not_exists(base_path):
    current_date = datetime.now().strftime('%Y%m%d')
    folder_path = os.path.join(base_path, current_date)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    return folder_path

def main():
    parser = argparse.ArgumentParser(description="Generations script with argparse")

    parser.add_argument('--prompt', type=str, default='User: Generate a very uplifting, inspiring, positive, continuation for the statement: ', help='Specify the prompt string.')
    parser.add_argument('--gen_name', type=str, default='toxic_to_benign', help='Specify the gen_name string. This serves as an identifier of what kind of generation you ran.')
    parser.add_argument('--model_name', type=str, default='mistralai/Mistral-7B-Instruct-v0.2', help='Specify the model name.')
    parser.add_argument('--prefix_file', type=str, default='/home/jsinha_umass_edu/Controlled-Text-Generation/data/toxic_prefixes.txt', help='Specify the prefix file.')
    parser.add_argument('--batch_size', type=int, default=4, help='Specify the batch size.')
    parser.add_argument('--max_new_tokens', type=int, default=60, help='Specify the max new tokens value.')
    parser.add_argument('--max_length', type=int, default=200, help='Specify the max length value.')
    parser.add_argument('--num_returns_sequences', type=int, default=2, help='Specify the num return sequences value.')
    parser.add_argument('--temperature', type=float, default=0.5, help='Specify the temperature value.')
    parser.add_argument('--top_p', type=float, default=0.95, help='Specify the top_p value.')
    parser.add_argument('--test', action='store_true', default=False, help='Specify if the script is to be run in Test mode. It will only run one iteration to check feasibility.')
    parser.add_argument('--no_save', action='store_true', default=False, help='Specify if the script is to be run in Test mode. It will only run one iteration to check feasibility.')

    pp = pprint.PrettyPrinter(indent=4)

    args = parser.parse_args()
    
    args_dict = vars(args)

    # Access the values of the arguments
    # prompt = args.prompt
    # gen_name = args.gen_name
    pp.pprint(args_dict)
    model_id = args.model_name.split('/')[0] + '-' + args.model_name.split('/')[1]
    device = torch.device("cuda")
    
    model, tokenizer = get_model_and_tokenizer(args.model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    prefs = open(args.prefix_file, encoding='UTF-8').readlines()
    prompt = args.prompt
    dataloader = get_dataloader(prefs, prompt, tokenizer, collate_fn=collate_fn, batch_size=args.batch_size, device = device, shuffle=True)
    
    completions = generate_completions(args, model, tokenizer, dataloader, args.batch_size, args.test)
    
    output_folder = create_folder_if_not_exists('/project/pi_mccallum_umass_edu/jsinha_umass_edu/Controlled-Text-Generation/output')
    current_time = datetime.now().time()
    time_string = current_time.strftime('%H%M%S')
    
    args_dict = vars(args)
    
    if not args.no_save:
        with open(output_folder + '/' + args.gen_name + "-args-" + time_string + ".json", 'w') as json_file:
            json.dump(args_dict, json_file, indent=2)
        dump_to_file(output_folder + '/' + args.gen_name + "-" + model_id + '-' + time_string + ".jsonl", prefs, completions)
    else:
        print(completions)

if __name__ == "__main__":
    main()