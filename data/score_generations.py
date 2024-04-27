# 1. Load the prefix and continuations files.
# 2. Load the SBERT model for computing the semantic textual similarity between the prefixes and continuations.
# 3. Compute the similarity scores between the prefixes and continuations.
# 4. Append back the similarity scores to the prefix and continuations files.
# 5. Save the updated prefix and continuations files.

import jsonlines
from sentence_transformers import SentenceTransformer, util
import torch
import numpy as np
model_name = 'all-MiniLM-L6-v2'
batch_size = 256

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_batches(prefixes, continuations, batch_size):
    """Generate batches of prefixes and continuations."""
    for i in range(0, len(prefixes), batch_size):
        # Ensure that the slicing does not go out of bounds
        batch_prefixes = prefixes[i:i + batch_size]
        batch_continuations = continuations[i:i + batch_size]
        yield (batch_prefixes, batch_continuations)

model = SentenceTransformer(model_name).to(device)


def sts_test(prefixes, continuations, model, result=[]):
    # Compute the embeddings for the prefixes and continuations
    prefix_embeddings = model.encode(prefixes, convert_to_tensor=True)
    continuation_embeddings = model.encode(continuations, convert_to_tensor=True)

    # Compute the cosine similarity scores
    cosine_scores = util.cos_sim(prefix_embeddings, continuation_embeddings)
    cosine_scores = np.diag(cosine_scores.cpu().numpy())
    for i in range(len(prefixes)):
        result.append({'prefix': prefixes[i], 'continuation': continuations[i], 'score': str(cosine_scores[i])})
    return result

def load_files(file_path):
    prefixes = []
    continuations = []
    with jsonlines.open(file_path, 'r') as reader:
        for line in reader:
            prefixes.append(line['prefix'])
            continuations.append(line['continuation'])
    return prefixes, continuations

data_files = ['../output/toxic_toxic_l3-meta-llama-Meta-Llama-3-8B-Instruct-233659.jsonl', '../output/toxic_toxic_l3-mistralai-Mistral-7B-Instruct-v0.2-002319.jsonl']

# Load the prefix and continuations files
for file_path in data_files:

    prefixes, continuations = load_files(file_path)
    result = []
    for batch_prefixes, batch_continuations in get_batches(prefixes, continuations, batch_size):
        result = sts_test(batch_prefixes, batch_continuations, model, result)

    # Write the updated prefix and continuations files
    with jsonlines.open(file_path, 'w') as writer:
        for line in result:
            writer.write(line)

