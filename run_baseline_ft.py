#!/usr/bin/env python
# coding: utf-8
import os
os.environ['HF_HOME'] = '/project/pi_mccallum_umass_edu/jsinha_umass_edu'
import warnings
warnings.filterwarnings('ignore')
import jsonlines
import json
import torch
import transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer, LogitsProcessor, AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler, AdamW
from datetime import datetime
import argparse
import pprint
from tqdm import tqdm
import wandb
import numpy as np
import torch.nn as nn
# Rough Outline
# 1. Load the model and tokenizer
# 2. Check the files in the training_files. Format: {prefix: str, "continuations": str}.
# 3. Create Dataset class and DataLoader. This is CausalLM, so the prefix and continuation are together one example. 
# 4. Define the collate_fn for the DataLoader. Keep in mind: Our total length is 316 or so after tokenization. So,
#    we need to pad to 512 with left_padding.
# 5. Define the training loop. As a test we will measure overfit on very small sample. This training loop will only train the 
#    expert model. We will freeze amateur model and the large one. 
# 6. Preferably use wandb to log the training.
# 7. As another baseline, we will train the amateur model functioning as an anti-expert too. -> This can be implemented as a separate
#    transformers model itself for it to be compatible with trl. 
# 8. Implement an eval set for the expert model. 
# 9. [Optional] Setup a sweep on wandb to find the best hyperparameters for the expert model and amateur models.


def create_folder_if_not_exists(base_path, job_name):
    current_date = datetime.now().strftime('%Y%m%d')
    folder_path = os.path.join(base_path, current_date)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    folder_path = os.path.join(folder_path, job_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path


# Define the dataset class
class PrefixDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prefix = []
        self.continuations = []
        with jsonlines.open(file_path, 'r') as reader:
            for line in reader:
                data = line
                prefix = data.get('prefix', '')
                continuations = data.get('continuation', '')
                text = prefix + ' ' + continuations
                self.prefix.append(text)
                self.continuations.append(text)
    def __len__(self):
        return len(self.prefix)

    def __getitem__(self, idx):
        return self.prefix[idx], self.continuations[idx]
      
# Tokenize dataset and create DataLoader
def collate_fn(batch, tokenizer):
    # Separate prefixes and continuations from the batch
    prefixes, continuations = zip(*batch)

    # Tokenize prefixes and continuations separately
    tokenized_prefixes = tokenizer(prefixes, return_tensors='pt', padding='max_length', truncation=True, max_length=150)
    tokenized_continuations = tokenizer(continuations, return_tensors='pt', padding='max_length', truncation=True, max_length=150)['input_ids']

    # Return input IDs and labels
    return tokenized_prefixes, tokenized_continuations

def train(args):
    # Load GPT-2 tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.padding_side = 'left'
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    large_model = AutoModelForCausalLM.from_pretrained(args.large_model_name).to(device)
    anti_expert_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)

    large_model.eval()

    if not args.train_anti_expert:
        anti_expert_model.eval()

    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    pp = pprint.PrettyPrinter(indent=4)
    args_dict = vars(args)

    pp.pprint(args_dict)
    
    tokenizer.pad_token = tokenizer.eos_token
    
    output_folder = create_folder_if_not_exists(args.output_path, args.job_name)
    current_time = datetime.now().time()
    time_string = current_time.strftime('%H%M%S')
    
    output_file_path = os.path.join(output_folder, 'train_' + time_string+'.jsonl')
    
    # Custom dataset
    dataset = PrefixDataset(args.training_file_path, tokenizer)
    args_dict['num_training_examples'] = len(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, tokenizer))

    # Fine-tuning
    
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.1)
    
    wandb.init(project='Guided-Text-Generation')
    # Optionally log hyperparameters
    wandb.config.update(args)
    
    for epoch in tqdm(range(args.epochs)):  # Set the number of epochs
        i = 1
        for batch, labels in tqdm(dataloader):
            input_ids = batch['input_ids'].to(device)
            input_attention_mask = batch['attention_mask'].to(device)
            labels = labels.to(device)


            scores = model(input_ids=input_ids, attention_mask=input_attention_mask, labels=labels).logits
            with torch.no_grad():
                large_model_logits = large_model(input_ids, attention_mask=input_attention_mask).logits
            
            if args.train_anti_expert:
                amateur_logits = anti_expert_model(input_ids, attention_mask=input_attention_mask).logits
            else:
                with torch.no_grad():
                    amateur_logits = anti_expert_model(input_ids, attention_mask=input_attention_mask).logits
                # Only take the last token of the input_ids
                # large_model_logits = large_model_logits[:, -1, :]
                # amateur_logits = amateur_logits[:, -1, :]

            cutoff = np.log(args.alpha) + large_model_logits.cpu().max(dim=-1, keepdim=True).values
            # diffs = large_model_logits + args.beta * scores - args.gamma * amateur_logits

            final_logits = large_model_logits + args.beta * scores - args.gamma * amateur_logits

            # final_logits = diffs.masked_fill(large_model_logits < cutoff.to(device), -float("inf"))

            # print("large_model_logits: ", large_model_logits[..., :-1, :])
            # print("amateur_logits: ", amateur_logits[..., :-1, :])
            # print("scores: ", scores[..., :-1, :])

            loss = None
            
            # move labels to correct device to enable model parallelism
            labels = labels.to(final_logits.device)
            # Shift so that tokens < n predict n
            shift_logits = final_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            

            i+=1

            wandb.log({"epoch": epoch, "loss": loss.item()})
            if i % 10 == 0 and args.verbose:
                print(f"Epoch: {epoch + 1}, Iteration: {i}, Loss: {loss.item()}")

        if args.test: 
            break
    
    if not args.no_save:
        with open(output_folder + '/' + args.job_name + "-args-" + time_string + ".json", 'w') as json_file:
            json.dump(args_dict, json_file, indent=2)
            
        model.save_pretrained(output_folder + '/' + args.model_name + '_' + time_string) 
        tokenizer.save_pretrained(output_folder + '/' + args.model_name + '_' + time_string)

        if args.train_anti_expert:
            anti_expert_model.save_pretrained(output_folder + '/' + args.model_name + '_anti_expert_' + time_string)
            tokenizer.save_pretrained(output_folder + '/' + args.model_name + '_anti_expert_' + time_string)

    wandb.finish()

def main():
    parser = argparse.ArgumentParser(description="Expert fine-tuning script with argparse.")

    parser.add_argument('--job_name', type=str, default='gpt_toxic_expert', help='Specify the job_name string. This serves as an identifier of what kind of training you ran.')
    parser.add_argument('--model_name', type=str, default='gpt2', help='Specify the model name. Default is GPT-2 Base.')
    parser.add_argument('--large_model_name', type=str, default='gpt2-large', help='Specify the model name. Default is GPT-2 Base.')
    parser.add_argument('--output_path', type=str, default='/project/pi_mccallum_umass_edu/jsinha_umass_edu/Controlled-Text-Generation/output', help='Specify the model name. Default is GPT-2 Base.')
    parser.add_argument('--training_file_path', type=str, default='/home/jsinha_umass_edu/Guided-Detoxification/output/', help='Specify the training files. These are .jsonl files that contain the synthetic data generations in a specific format.')
    parser.add_argument('--batch_size', type=int, default=4, help='Specify the batch size. Default: 4')
    parser.add_argument('--epochs', type=int, default=5, help='Specify the number of epochs. Default: 5')
    parser.add_argument('--step_size', type=int, default=200, help='Specify the step size. Default: 1')
    parser.add_argument('--seed', type=int, default=42, help='Specify the seed. Default: 42')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Specify the learning rate. Default: 1e-4')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Specify weight decay. Default: 0.01')
    parser.add_argument('--gamma', type=float, default=0.8, help='Specify the gamma for anti_expert_model. Default: 0.8')
    parser.add_argument('--alpha', type=float, default=0.1, help='Specify the gamma for anti_expert_model. Default: 0.1')
    parser.add_argument('--beta', type=float, default=0.6, help='Specify the gamma for anti_expert_model. Default: 0.6')
    parser.add_argument('--train_anti_expert', action='store_true', default=False, help='Specify if we should train the anti-expert too.')
    parser.add_argument('--no_save', action='store_true', default=False, help='Specify if the script is to be run in Test mode. It will only run one iteration to check feasibility.')
    parser.add_argument('--test', action='store_true', default=False, help='Specify if the script is to be run in Test mode. It will only run one iteration to check feasibility.')
    parser.add_argument('--verbose', action='store_true', default=False, help='Specify if the script is to be run in verbose mode. If set, this will display the loss at each 20 iteration step.')
   

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    transformers.set_seed(args.seed)
    
    
    train(args)
    
    
    
    
if __name__ == "__main__":
    main()