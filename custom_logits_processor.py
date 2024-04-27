#!/usr/bin/env python3
from transformers import LogitsProcessor
import torch
import numpy as np

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