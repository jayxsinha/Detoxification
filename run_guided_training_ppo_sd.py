# Reference: https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/gpt-j-6b-toxicity.py

# TODO: Implement DPOTrainer for training the model.
# TODO: Implement synthetic data generation script for training the model.

import os
os.environ['HF_HOME'] = '/project/pi_mccallum_umass_edu/jsinha_umass_edu'
WANDB_ENTITY="jaysinha"
WANDB_PROJECT="Guided-Detoxification"
os.environ['WANDB_ENTITY'] = WANDB_ENTITY
os.environ['WANDB_PROJECT'] = WANDB_PROJECT
from dataclasses import dataclass, field
from typing import Optional
import torch
from datasets import load_dataset, Dataset
from torch.optim import Adam
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    RobertaForSequenceClassification,
    RobertaTokenizer,
    LogitsProcessorList,
    LogitsProcessor,
)
import numpy as np
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed
from trl.core import LengthSampler
import jsonlines

########################################################################
class ExpertOnlyLogitsProcessor(LogitsProcessor):
    """
    This processor uses the expert only to guide the generations of the larger model.
    """
    def __init__(self, large_model, alpha=0.1, beta=0.6):
        self.alpha = alpha
        self.beta = beta
        self.large_model = large_model


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        large_model_logits = self.large_model(input_ids).logits
        # Only take the last token of the input_ids
        with torch.no_grad():
            large_model_logits = large_model_logits[:, -1, :]
        cutoff = np.log(self.alpha) + large_model_logits.cpu().max(dim=-1, keepdim=True).values
        diffs = large_model_logits + self.beta * scores
        final_logits = diffs.masked_fill(large_model_logits < cutoff.to(device), -float("inf"))
        return final_logits       
    
class ExpertAmateurLogitsProcessor(LogitsProcessor):
    """
    This processor uses the expert and amateur both to guide the generations of the larger model.
    """

    def __init__(self, large_model, amateur_model, tokenizer, alpha=0.1, beta=0.6, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.large_model = large_model
        self.amateur_model = amateur_model
        self.tokenizer = tokenizer

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:

        with torch.no_grad():
            large_model_logits = self.large_model(input_ids).logits
            amateur_logits = self.amateur_model(input_ids).logits
        # Only take the last token of the input_ids
        large_model_logits = large_model_logits[:, -1, :]
        amateur_logits = amateur_logits[:, -1, :]

        cutoff = np.log(self.alpha) + large_model_logits.cpu().max(dim=-1, keepdim=True).values
        diffs = large_model_logits + self.beta * scores - self.gamma * amateur_logits
        final_logits = diffs.masked_fill(large_model_logits < cutoff.to(device), -float("inf"))
        

        # Run the final_logits and print the next token
        next_token_logits = final_logits[0]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        next_token_id = torch.argmax(next_token_probs).item()
        next_token = self.tokenizer.decode(next_token_id)
        print("Input: ", self.tokenizer.decode(input_ids[0]))
        print("Next token: ", next_token)


        return final_logits   
########################################################################

tqdm.pandas()

# create a const variable to store current date and time in the format = 'YYYYMMDD-HHMM' by using the datetime module
import datetime
now = datetime.datetime.now()
current_time = now.strftime("%Y%m%d-%H%M")
model_save_path = '/project/pi_mccallum_umass_edu/jsinha_umass_edu/ctg-detox-' + current_time

device = "cuda" if torch.cuda.is_available() else "cpu"

########################################################################
# This is a fully working simple example to use trl with accelerate.
#
# This example fine-tunes a GPTJ model to generate less toxic contents
# by using allenai/real-toxicity-prompts dataset. We use PPO
#  (proximal policy optimization) to optimize the model.
# in any of the following settings (with the same script):
#   - single CPU or single GPU
#   - multi GPUS (using PyTorch distributed mode)
#   - multi GPUS (using DeepSpeed ZeRO-Offload stages 1 & 2)
#   - fp16 (mixed-precision) or fp32 (normal precision)
#
# To run it in each of these various modes, first initialize the accelerate
# configuration with `accelerate config`
#
########################################################################


# We first define the configuration of the experiment, defining the model, the dataset,
# the training parameters, and the PPO parameters.
# Check the default arguments in the `PPOConfig` class for more details.
# If you want to log with tensorboard, add the kwarg
# `project_kwargs={"logging_dir": PATH_TO_LOGS}` to the PPOConfig.
@dataclass
class ScriptArguments:
    """
    The name of the Casual LM model we wish to fine-tune with PPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    large_model_name: Optional[str] = field(default="gpt2-large", metadata={"help": "the large model name"})
    amateur_model_name: Optional[str] = field(default="gpt2", metadata={"help": "the amateur model name"})
    log_with: Optional[str] = field(default=None, metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=32, metadata={"help": "the PPO minibatch size"})
    batch_size: Optional[int] = field(default=64, metadata={"help": "the batch size"})
    steps: Optional[int] = field(default=None, metadata={"help": "the number of training steps"})
    ppo_epochs: Optional[int] = field(default=4, metadata={"help": "the number of PPO epochs"})
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    # Model save path should be appended with date and time
    model_save_path: Optional[str] = field(
        default=model_save_path,
        metadata={"help": "the path to save the model"},
    )
    guidance_mode: Optional[str] = field(
        default="expert_amateur",
    )
    alpha: Optional[float] = field(
        default=0.1,
    )
    beta: Optional[float] = field(
        default=0.6,
    )
    gamma: Optional[float] = field(
        default=0.8,
    )
    training_type: Optional[str] = field(
        default="ppo",
    )




parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]


config = PPOConfig(
    model_name=script_args.model_name,
    learning_rate=script_args.learning_rate,
    log_with=script_args.log_with,
    ppo_epochs=script_args.ppo_epochs,
    mini_batch_size=script_args.mini_batch_size,
    batch_size=script_args.batch_size,
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
)


# Below is an example function to build the dataset. In our case, we use the IMDB dataset
# from the `datasets` library. One should customize this function to train the model on
# its own dataset.
def build_dataset(
    config, ds, input_min_text_length=5, input_max_text_length=10
):
    """
    Build dataset for training. This builds the dataset from `load_dataset`, one should
    customize this function to train the model on its own dataset.

    Args:
        ds (`list[dict{'prefix': str, 'continuation': str, 'score': float}]`):
            The data to be loaded.

    Returns:
        dataloader (`torch.utils.data.DataLoader`):
            The dataloader for the dataset.
    """
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # ds = load_dataset(dataset_name, split="train")

    # def filter_fn(sample):
        # toxicity = sample["prompt"]["toxicity"]
        # return toxicity is not None and toxicity > 0.3

    # ds = ds.filter(filter_fn, batched=False)

    input_size = LengthSampler(input_min_text_length, input_max_text_length)

    def tokenize(sample):
        prompt = sample["prompt"]
        continuation = sample["continuation"]

        sample["input_ids"] = tokenizer.encode(prompt + continuation)[: input_size()]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample

    # result = {'prompt': [], 'continuation': [], 'input_ids': [], 'query': []}
    result = {'prompt': [], 'continuation': []}
    for sample in ds:
        # tkd = tokenize(sample)
        result['prompt'].append(sample['prefix'])
        result['continuation'].append(sample['continuation'])
        # result['input_ids'].append(sample['input_ids'])
        # result['query'].append([sample['query']])

    ds = Dataset.from_dict(result)
    
    ds = ds.map(tokenize, batched=False)
    ds.set_format(type="torch")

    ds = ds.train_test_split(test_size=0.2, shuffle=False)["train"]

    return ds

# Get the dataset from: output/toxic_to_benign_20240425.jsonl
# The dataset is a jsonl file with the following format:
# {"prefix": "prefix text", "continuation": "continuation text", "score": "score"}

data = []
with jsonlines.open('output/ppo_train_data_20240427.jsonl', 'r') as reader:
    data = [line for line in reader]


# We retrieve the dataloader by calling the `build_dataset` function.
min_input_length = 20
max_input_length = 1024
# print("len(data): ", len(data))
dataset = build_dataset(config, ds=data, input_min_text_length=min_input_length, input_max_text_length=max_input_length)

# print(dataset)
# exit(0)

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}



# set seed before initializing value head for deterministic eval
set_seed(config.seed)

# Now let's build the model, the reference model, and the tokenizer. We first load the model
# in bfloat16 to save memory using `transformers`.
model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.bfloat16)
# And then we pass the loaded model to `AutoModelForCausalLMWithValueHead`.
model = AutoModelForCausalLMWithValueHead.from_pretrained(model).to(device)

# We create a reference model by sharing 12 layers
ref_model = create_reference_model(model, num_shared_layers=6).to(device)

# We make sure to use `Adam` optimizer on the model parameters that require gradients.
optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.learning_rate)

# GPT-2 / GPT-J tokenizer has a pad token, but it is not eos_token by default. We need to set it to eos_token.
# only for this model.
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

if script_args.guidance_mode == "expert_only":
    large_model = AutoModelForCausalLM.from_pretrained(script_args.large_model_name).to(device)
    large_model = large_model.eval()
    logits_processor = ExpertOnlyLogitsProcessor(large_model, alpha=script_args.alpha, beta=script_args.beta)
else:
    large_model = AutoModelForCausalLM.from_pretrained(script_args.large_model_name).to(device)
    large_model = large_model.eval()
    amateur_model = AutoModelForCausalLM.from_pretrained(script_args.amateur_model_name).to(device)
    amateur_model = amateur_model.eval()
    logits_processor = ExpertAmateurLogitsProcessor(large_model, amateur_model, tokenizer=tokenizer, alpha=script_args.alpha, beta=script_args.beta, gamma=script_args.gamma)


# We then build the PPOTrainer, passing the model, the reference model, the tokenizer
ppo_trainer = PPOTrainer(
    config,
    model,
    ref_model=ref_model,
    tokenizer=tokenizer,
    dataset=dataset,
    data_collator=collator,
    optimizer=optimizer,
)

# We then build the reward pipeline, we will use the toxicity model to compute the reward.
# We first load the toxicity model and tokenizer.
toxicity_model_id = "facebook/roberta-hate-speech-dynabench-r4-target"
toxicity_tokenizer = RobertaTokenizer.from_pretrained(toxicity_model_id)
# We load the toxicity model in fp16 to save memory.
toxicity_model = RobertaForSequenceClassification.from_pretrained(toxicity_model_id, torch_dtype=torch.float16).to(
    ppo_trainer.accelerator.device
)


# We then define the arguments to pass to the `generate` function. These arguments
# are passed to the `generate` function of the PPOTrainer, which is a wrapper around
# the `generate` function of the trained model.
generation_kwargs = {
    "min_length": -1,
    "max_new_tokens": 20,
    "top_k": 20,
    "top_p": 0.9,
    "do_sample": True,
    "pad_token_id": tokenizer.eos_token_id,
    "logits_processor": LogitsProcessorList([logits_processor]),
    "num_return_sequences": 1,
}

output_min_length = 20
output_max_length = 30
output_length_sampler = LengthSampler(output_min_length, output_max_length)

model_save_path = script_args.model_save_path

for epoch, batch in tqdm(enumerate(ppo_trainer.dataloader)):
    # print(batch)
    query_tensors = batch["input_ids"]
    # Get response from the policy model
    response_tensors = []
    for query in query_tensors:
        gen_len = output_length_sampler()
        generation_kwargs["max_new_tokens"] = gen_len
        response = ppo_trainer.generate(query, **generation_kwargs)
        response_tensors.append(response.squeeze()[-gen_len:])
    batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]

    # Compute sentiment score
    texts = batch["response"]
    toxicity_inputs = toxicity_tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(
        ppo_trainer.accelerator.device
    )
    logits = toxicity_model(**toxicity_inputs).logits.float()
    toxicity_labels = (logits[:, 0]).tolist()

    rewards = [torch.tensor(output) for output in toxicity_labels]

    # Run PPO step
    stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
    exit(0)
    ppo_trainer.log_stats(stats, batch, rewards)

    # Save model every 100 epochs
    # if epoch % 1 == 0:
    if ppo_trainer.accelerator.is_main_process:
        ppo_trainer.save_pretrained(model_save_path)