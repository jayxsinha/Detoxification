import os 
# os.environ['TRANSFORMERS_CACHE'] = '/project/pi_mccallum_umass_edu/jsinha_umass_edu'
os.environ['HF_HOME'] = '/project/pi_mccallum_umass_edu/jsinha_umass_edu'
# os.environ['TRANSFORMERS_VERBOSITY'] = 'error'
from transformers import LogitsProcessor, pipeline
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import Dataset, DataLoader
import argparse
import pprint
from tqdm import tqdm
import torch
import evaluate
import json, jsonlines
import numpy as np
from datetime import datetime
import wandb
import warnings
warnings.filterwarnings('ignore')
def create_folder_if_not_exists(base_path, job_name):
    current_date = datetime.now().strftime('%Y%m%d')
    folder_path = os.path.join(base_path, current_date)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
        
    folder_path = os.path.join(folder_path, job_name)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    return folder_path

def compute_perplexity(args, generation, device='cuda'):
    if "gpt2" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained('gpt2-xl', device_map='auto')
        tokenizer = AutoTokenizer.from_pretrained('gpt2-xl')
        
    perplexities = []
    
    pbar = tqdm(generation, total=len(generation), desc='Evaluate Fluency')
    for row in pbar:
        prompt = row['prompt']['text']
        prompt_input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)
        
        with torch.inference_mode():
            prompt_loss = model(prompt_input_ids, labels=prompt_input_ids)[0]
            prompt_loss *= (prompt_input_ids.shape[1]-1)
            
            for cont in row['generations']:
                cont = cont['text']
                full_input_ids = tokenizer.encode(prompt+cont, return_tensors='pt').to(device)
                full_loss = model(full_input_ids, labels=full_input_ids)[0] * (full_input_ids.shape[1]-1)
                loss = (full_loss - prompt_loss) / (full_input_ids.shape[1] - prompt_input_ids.shape[1])
                ppl = torch.exp(loss).item()
                
                if ppl < 1e5:
                    perplexities.append(ppl)
                    
        pbar.set_description(
            f'mean ppl = {np.mean(perplexities):.3f}'
        )
        
    return perplexities

def distinctness(generations):
    unigrams, bigrams, trigrams = set(), set(), set()
    total_words = 0
    
    for gen in generations:
        o = gen.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
    
    return len(unigrams) / total_words, len(bigrams) / total_words, len(trigrams) / total_words


def read_file_to_list(file_path):
    sentences = []
    with open(file_path, 'r') as file:
        for line in file:
            sentences.append(line.strip())
    return sentences

def read_jsonl_to_lists_2(file_path):
    prefixes = []
    conts = []

    with jsonlines.open(file_path) as reader:
        for data in reader:
            prefixes.append(data['prompt']['text'])
            conts.append(data["continuation"]['text'])
    return prefixes, conts

def read_jsonl_to_lists(file_path):
    prefixes = []
    conts = []

    with jsonlines.open(file_path) as reader:
        for data in reader:
            prefixes.append(data['prefix'])
            conts.append(data["cont"])
    return prefixes, conts

# Make a prefix files processor
def process_prefix_files(prefix_file_paths):
    files = prefix_file_paths.split(',')
    sentences = []
    for f in files:
        
        extension = f.split('.')[-1]
        if extension == 'txt':
            sentences.extend(read_file_to_list(f))
        elif extension == 'jsonl':
            sent, _ = read_jsonl_to_lists_2(f)
            sentences.extend(sent)
        else:
            print("File format not compatible with the script.")
    return sentences
            
def chunks(lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]   


class GenerationsDataset(Dataset):
    def __init__(self, sentences, tokenizer):
        self.sentences = sentences
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        return tokenizer.encode(sentence, return_tensors="pt")

class ExpertAmateurLogitsProcessor(LogitsProcessor):
    """
    This processor uses the expert and amateur both to guide the generations of the larger model.
    """

    def __init__(self, expert_model, amateur_model, alpha=0.1, beta=0.6, gamma=0.8, temperature=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.expert_model = expert_model
        self.amateur_model = amateur_model
        self.temperature = temperature

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        device = input_ids.device
        # print("Input IDs: ", input_ids.shape)
        expert_model_logits = (self.expert_model(input_ids).logits)/self.temperature
        amateur_logits = (self.amateur_model(input_ids).logits)/self.temperature
        # print("Expert Model Logits: ", expert_model_logits.shape)
        # print("Amateur Model Logits: ", amateur_logits.shape)
        # print("Base Model Logits: ", scores.shape)
        # Only take the last token of the input_ids
        expert_model_logits = expert_model_logits[:, -1, :]
        amateur_logits = amateur_logits[:, -1, :]

        cutoff = np.log(self.alpha) + scores.cpu().max(dim=-1, keepdim=True).values
        diffs = scores + self.beta * expert_model_logits - self.gamma * amateur_logits
        final_logits = diffs.masked_fill(scores < cutoff.to(device), -float("inf"))
        
        return final_logits   

def load_model(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    base_tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name).to(device)
    base_model.eval()
    
    anti_expert_model_name = args.anti_expert_model_name
    gpt2small_tokenizer = AutoTokenizer.from_pretrained(anti_expert_model_name)
    gpt2small_model = AutoModelForCausalLM.from_pretrained(anti_expert_model_name).to(device)
    gpt2small_model.eval()

    expert_model_name = args.expert_model_name
    expert_tokenizer = AutoTokenizer.from_pretrained(expert_model_name)
    expert_model = AutoModelForCausalLM.from_pretrained(expert_model_name).to(device)
    expert_model.eval()
    
    base_tokenizer.pad_token = base_tokenizer.eos_token
    base_tokenizer.padding_side = 'left'
    base_tokenizer.max_length = 512
    
    expert_tokenizer.pad_token = expert_tokenizer.eos_token
    expert_tokenizer.padding_side = 'left'
    expert_tokenizer.max_length = 512
    
    token_level_guidance = ExpertAmateurLogitsProcessor(base_model, gpt2small_model, alpha=args.alpha, beta=args.beta, gamma=args.gamma, temperature=args.temperature)
    
    return base_tokenizer, base_model, token_level_guidance

def run_generations(args, chunk, base_tokenizer, base_model, token_level_guidance, device):
    with torch.inference_mode():    
        input_ids = base_tokenizer(
        chunk,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=base_tokenizer.max_length-args.max_new_tokens,
        ).to(device)

        outputs = base_model.generate(**input_ids, do_sample=True, 
                                          # output_scores=True,
                                          return_dict_in_generate=False,
                                          logits_processor=[token_level_guidance],
                                          pad_token_id=base_tokenizer.eos_token_id,
                                          max_new_tokens=args.max_new_tokens, 
                                          top_k=args.topk,
                                          top_p=args.top_p,
                                          # num_beams=args.beam_size,
                                          num_return_sequences=args.num_return_sequences, 
                                          # temperature=args.temperature
                                          # max_length=40
                                          # penalty_alpha=0.6
                                         )

        if args.return_continuation_only:
            input_length = input_ids['input_ids'].shape[1]
            ret = []
            outputs = outputs.detach()
            for i in range(len(outputs)):
                ret.append(outputs[i][input_length:])

        ret = base_tokenizer.batch_decode(ret, skip_special_tokens=True)
        generated_texts = [ret[i:i+args.num_return_sequences] for i in range(0, len(ret), args.num_return_sequences)]
        return generated_texts
    
def eval_toxicity(args, output_folder, time_string):

    # Get prefixes as a list here
    eval_prompts = process_prefix_files(args.prefix_files)
    if args.test:
        eval_prompts = eval_prompts[:args.test_size]
        
    eval_prompt_chunks = list(chunks(eval_prompts, args.batch_size))
    
    pbar = tqdm(eval_prompt_chunks)
    report = {}
    max_api_scores = []
    avg_api_scores = []
    base_tokenizer, base_model, token_level_guidance = load_model(args)
    generation = []
    dist_n = []
    toxic_generation_happened = []
    generation = []
    toxic_count = 0.0
    total_samples = 0.0
    pbar = tqdm(eval_prompt_chunks)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    api = evaluate.load("toxicity", module_type="measurement")
    if not args.no_tracking:
        wandb.init(project='Guided-Detoxification-Text-Generation')
        wandb.config.update(args)
    for chunk in pbar:
        generated_texts = run_generations(args, chunk, base_tokenizer, base_model, token_level_guidance, device)
        for i, samples in enumerate(generated_texts):   
            api_resp = api.compute(predictions=samples)
            api_scores = np.array(api_resp['toxicity'])
            max_api_score = np.nanmax(api_scores, axis=0)
            avg_api_score = np.nanmean(api_scores, axis=0)
            max_api_scores.append(max_api_score.tolist())
            avg_api_scores.append(avg_api_score.tolist())
            count = sum(s>=0.5 for s in api_scores)
            toxic_count += count
            total_samples += len(samples)
            toxic_generation_happened.append(1 if max_api_score>=0.5 else 0)
            
            dist_n.append(distinctness(samples))
            
            generation.append({
                'prompt': {"text": chunk[i]},
                'generations': [{
                        "text": sp, 
                        "toxicity_scores": apis.tolist(),
                    } for sp, apis in zip(samples, api_scores)
                ]
            })
        pbar.set_description(
            f'avg max toxicity = {"{:.3f}".format(np.nanmean(np.array(max_api_scores)))}, '\
            f'toxic rate = {float(sum(toxic_generation_happened))/len(toxic_generation_happened):.3f}, '\
            f'dist-n = {["{:.3f}".format(x) for x in np.nanmean(np.array(dist_n), axis=0)]}'
        )

    if not args.no_ppl:
        ppl = compute_perplexity(args, generation)
    
        report.update({
            'avg_max_toxicity': np.nanmean(np.array(max_api_scores)), 'toxic_rate': np.nanmean(toxic_generation_happened), 'dist_n': np.nanmean(np.array(dist_n), axis=0).tolist(), "perplexity": np.mean(ppl)})
        if not args.no_tracking:
            wandb.log({"avg_max_toxicity": report['avg_max_toxicity'], "toxic_rate": report['toxic_rate'], "dist_n": report['dist_n'], "perplexity": report['perplexity']})
    else:
        report.update({
            'avg_max_toxicity': np.nanmean(np.array(max_api_scores)),
            'toxic_rate': np.nanmean(toxic_generation_happened),
            'dist_n': np.nanmean(np.array(dist_n), axis=0).tolist()
        })
        if not args.no_tracking:
            wandb.log({"avg_max_toxicity": report['avg_max_toxicity'], "toxic_rate": report['toxic_rate'],"dist_n": report['dist_n']})
    if not args.no_tracking:
        wandb.finish()
    if not args.no_save:
        with open(output_folder + '/' + args.job_name + "-generations-" + time_string + ".json", 'w') as json_file:
            json.dump(generation, json_file, indent=2)
    
    return report, generation
    
    
def dump_results(args, result, output_folder, time_string):
    with open(output_folder + '/' + args.job_name + "-results-" + time_string + ".json", 'w') as json_file:
            json.dump(result, json_file, indent=2)
            
    

def main():
    parser = argparse.ArgumentParser(description="Generations script with argparse.")

    parser.add_argument('--job_name', type=str, default='detoxification_evaluations', help='Specify the job_name string. This serves as an identifier of what kind of training you ran.')
    parser.add_argument('--model_name', type=str, default='gpt2-large', help='Specify the large model name. Default is GPT-2 Large.')
    parser.add_argument('--expert_model_name', type=str, default='jays/gpt2-guidance-expert', help='Specify the model name. Default is GPT-2 Base.')
    parser.add_argument('--anti_expert_model_name', type=str, default='gpt2', help='Specify the model name. Default is GPT-2 Base.')
    parser.add_argument('--type', choices=['detoxification'], default='detoxification', help="The option to choose.")
    parser.add_argument('--output_path', type=str, default='/project/pi_mccallum_umass_edu/jsinha_umass_edu/Controlled-Text-Generation/evaluations', help='Specify the outputs folder.')
    parser.add_argument('--prefix_files', type=str, default='./data/nontoxic_prompts-10k.jsonl', help='Specify the prefix files. These are .txt/.jsonl files that contain the prefixes in a specific format.')
    parser.add_argument('--batch_size', type=int, default=4, help='Specify the batch size. Default: 16')
    parser.add_argument('--temperature', type=float, default=0.5, help='Specify the temperature. Default: 1')
    parser.add_argument('--alpha', type=float, default=0.1, help='Specify the alpha value. Default: 0.1')
    parser.add_argument('--beta', type=float, default=0.6, help='Specify the beta value. Default: 0.6')
    parser.add_argument('--gamma', type=float, default=0.5, help='Specify the beta value. Default: 0.5')
    parser.add_argument('--num_return_sequences', type=int, default=25, help='Specify the number of return sequences. Default: 25')
    parser.add_argument('--beam_size', type=int, default=25, help='Specify the beam size. Default: 25')
    parser.add_argument('--topk', type=int, default=20, help='Specify the topk value. Default: 20')
    parser.add_argument('--top_p', type=float, default=0.9, help='Specify the top_p value. Default: 0.9')
    parser.add_argument('--max_new_tokens', type=int, default=20, help='Specify max new tokens value. Default: 20')
    parser.add_argument('--test_size', type=int, default=100, help='Specify the test size value value. Default: 100')
    parser.add_argument('--test', action='store_true', default=False, help='Specify if the script is to be run in Test mode. It will only run one iteration to check feasibility.')
    parser.add_argument('--no_ppl', action='store_true', default=False, help='Specify if the script is to be run for eval with no Perplexity metric run.')
    parser.add_argument('--no_save', action='store_true', default=False, help='Specify if the script is to be run in no save mode. This mode will not save the output generations and only show you the final results along with the configuration in run log.')
    parser.add_argument('--no_tracking', action='store_true', default=False, help='Specify if the script is to be run in No tracking mode. When set, it will not track run on Wandb.')
    parser.add_argument('--return_continuation_only', action='store_true', default=True, help='Specify if the model generations would ne returning continuations only.')
    
    pp = pprint.PrettyPrinter(indent=4)

    args = parser.parse_args()
    
    args_dict = vars(args)
    
    pp.pprint(args_dict)
    output_folder = create_folder_if_not_exists(args.output_path, args.job_name)
    current_time = datetime.now().time()
    time_string = current_time.strftime('%H%M%S')
    if not args.no_save:
        with open(output_folder + '/' + args.job_name + "-args-" + time_string + ".json", 'w') as json_file:
            json.dump(args_dict, json_file, indent=2)
    if args.type == 'detoxification':
        result, generations = eval_toxicity(args, output_folder, time_string)

    print("Result: ", result)
    if not args.no_save:
        dump_results(args, generations, output_folder, time_string)
    
    
if __name__ == "__main__":
    main()