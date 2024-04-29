# Reference: https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/gpt-j-6b-toxicity.py

# TODO: Implement DPOTrainer for training the model.
# TODO: Implement synthetic data generation script for training the model.

import os
# os.environ['HF_HOME'] = '/project/pi_mccallum_umass_edu/jsinha_umass_edu'
# WANDB_ENTITY="jaysinha"
# WANDB_PROJECT="Guided-Detoxification"
# os.environ['WANDB_ENTITY'] = WANDB_ENTITY
# os.environ['WANDB_PROJECT'] = WANDB_PROJECT
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
from trl import AutoModelForCausalLMWithValueHead, PPOConfig, PPOTrainer, create_reference_model, set_seed, DPOConfig, DPOTrainer
from trl.core import LengthSampler
import json

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

    def __init__(self, large_model, amateur_model, alpha=0.1, beta=0.6, gamma=0.5):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.large_model = large_model
        self.amateur_model = amateur_model

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
        
        return final_logits   
########################################################################

tqdm.pandas()

# create a const variable to store current date and time in the format = 'YYYYMMDD-HHMM' by using the datetime module
import datetime
now = datetime.datetime.now()
current_time = now.strftime("%Y%m%d-%H%M")
# model_save_path = '/project/pi_mccallum_umass_edu/jsinha_umass_edu/ctg-detox-' + current_time
model_save_path = './model/ctg-detox-' + current_time

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
    The name of the Casual LM model we wish to fine-tune with DPO
    """

    # NOTE: gpt2 models use Conv1D instead of Linear layers which are not yet supported in 8 bit mode
    # models like gpt-neo* models are more suitable.
    model_name: Optional[str] = field(default="gpt2", metadata={"help": "the model name"})
    large_model_name: Optional[str] = field(default="EleutherAI/gpt-neo-125m", metadata={"help": "the large model name"})
    amateur_model_name: Optional[str] = field(default="gpt2", metadata={"help": "the amateur model name"})
    log_with: Optional[str] = field(default='wandb', metadata={"help": "use 'wandb' to log with wandb"})
    learning_rate: Optional[float] = field(default=(1.47e-5) * 2, metadata={"help": "the learning rate"})
    mini_batch_size: Optional[int] = field(default=64, metadata={"help": "the DPO minibatch size"})
    batch_size: Optional[int] = field(default=128, metadata={"help": "the batch size"})
    steps: Optional[int] = field(default=None, metadata={"help": "the number of training steps"})
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
        default="dpo",
    )




parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses(return_remaining_strings=True)[0]



if script_args.guidance_mode == "expert_only":
    large_model = AutoModelForCausalLM.from_pretrained(script_args.large_model_name).to(device)
    large_model = large_model.eval()
    logits_processor = ExpertOnlyLogitsProcessor(large_model)
else:
    large_model = AutoModelForCausalLM.from_pretrained(script_args.large_model_name).to(device)
    large_model = large_model.eval()
    amateur_model = AutoModelForCausalLM.from_pretrained(script_args.amateur_model_name).to(device)
    amateur_model = amateur_model.eval()
    logits_processor = ExpertAmateurLogitsProcessor(large_model, amateur_model)


def init_dummy_dataset():
    with open("./data/dpo_dataset.json", "r") as f:
        dummy_dataset_dict = json.load(f)
    return Dataset.from_dict(dummy_dataset_dict["dummy_dataset_dict"])

# Test the function
dataset = init_dummy_dataset()


# print(dataset[1])
model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
ref_model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
tokenizer.pad_token = tokenizer.eos_token


def calculate_max_length(dataset):
    max_input_length = max(len(tokenizer.encode(prompt)) for prompt in dataset["prompt"])
    max_output_length = max(max(len(tokenizer.encode(chosen)), len(tokenizer.encode(rejected))) for chosen, rejected in zip(dataset["chosen"], dataset["rejected"]))
    max_prompt_length = max(max_input_length, max_output_length)
    max_length = max_input_length + max_output_length  
    return max_length, max_prompt_length

# Calculate max_length and max_prompt_length
max_length, max_prompt_length = calculate_max_length(dataset)

print("Max Length:", max_length)
print("Max Prompt Length:", max_prompt_length)


model = AutoModelForCausalLM.from_pretrained(script_args.model_name)
ref_model = AutoModelForCausalLM.from_pretrained(script_args.model_name)

training_args = DPOConfig(
                    output_dir=model_save_path,
                    per_device_train_batch_size=2,
                    max_steps=10,
                    remove_unused_columns=False,
                    gradient_accumulation_steps=1,
                    learning_rate=script_args.learning_rate,
                    evaluation_strategy="steps",
                    beta=script_args.beta,
                    max_prompt_length=max_prompt_length,
                    max_length=max_length,
                    fp16=True,
                    logging_strategy="no",
                    report_to="none",
                )



trainer = DPOTrainer(
            model=model,
            ref_model=ref_model,
            args=training_args,
            tokenizer=tokenizer,
            train_dataset=dataset,
            eval_dataset=dataset,
        )

previous_trainable_params = {n: param.clone() for n, param in trainer.model.named_parameters()}

trainer.train()
for step, log_history in enumerate(trainer.state.log_history):
    print(f"Step: {step}, Train Loss: {log_history['train_loss']}")


assert trainer.state.log_history[-1]["train_loss"] is not None

# check the params have changed
for n, param in previous_trainable_params.items():
    new_param = trainer.model.get_parameter(n)
    # check the params have changed - ignore 0 biases
    if param.sum() != 0:
        assert not torch.equal(param, new_param)


trainer.save_model(training_args.output_dir)


