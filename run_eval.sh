#!/bin/bash

python run_eval.py --test --gamma=0.5 --beta=0.6 --test_size=128 --no_save --no_ppl --no_tracking --batch_size=16 \
--expert_model_name='/project/pi_mccallum_umass_edu/jsinha_umass_edu/Controlled-Text-Generation/output/20240426/gpt_toxic_expert/gpt2_224721' \
--anti_expert_model_name='/project/pi_mccallum_umass_edu/jsinha_umass_edu/Controlled-Text-Generation/output/20240426/gpt_toxic_expert/gpt2_anti_expert_224721'