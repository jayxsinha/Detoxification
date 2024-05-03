#!/bin/bash

python run_eval.py --alpha=0.1 --gamma=0.8 --beta=0.6  --no_ppl --batch_size=16 --topk=50 \
--expert_model_name='/project/pi_mccallum_umass_edu/jsinha_umass_edu/ctg-detox-20240430-0240'
# --anti_expert_model_name='/project/pi_mccallum_umass_edu/jsinha_umass_edu/Controlled-Text-Generation/output/20240426/gpt/_toxic_expert/gpt2_anti_expert_224721'