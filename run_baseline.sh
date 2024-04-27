#!/bin/bash

# python run_baseline_ft.py --no_save --training_file_path='output/toxic_to_benign_20240425.jsonl' --batch_size=32 --verbose
# python run_baseline_ft.py --training_file_path='output/toxic_to_benign_20240425.jsonl' --gamma=0.5 --batch_size=32 --learning_rate=1e-4 --epochs=2
# python run_baseline_ft.py --no_save --training_file_path='output/toxic_to_benign_20240425.jsonl' --gamma=0.2 --batch_size=32 --learning_rate=1e-4 --epochs=2
# python run_baseline_ft.py --no_save --training_file_path='output/toxic_to_benign_20240425.jsonl' --gamma=0.75 --batch_size=32 --learning_rate=1e-4 --epochs=2


# python run_baseline_ft.py --no_save --training_file_path='output/toxic_to_benign_20240425.jsonl' --gamma=0.5 --beta=1.0 --batch_size=32 --learning_rate=1e-4 --epochs=2 --test --train_anti_expert
# python run_baseline_ft.py --no_save --training_file_path='output/toxic_to_benign_20240425.jsonl' --gamma=0.75 --beta=1.0 --batch_size=32 --learning_rate=1e-4 --epochs=2 --test --train_anti_expert
# python run_baseline_ft.py --no_save --training_file_path='output/toxic_to_benign_20240425.jsonl' --gamma=0.25 --beta=1.0 --batch_size=32 --learning_rate=1e-4 --epochs=2 --test --train_anti_expert


# python run_baseline_ft.py --no_save --training_file_path='output/toxic_to_benign_20240425.jsonl' --gamma=0.5 --beta=0.5 --batch_size=32 --learning_rate=1e-4 --epochs=2 --test --train_anti_expert
python run_baseline_ft.py --training_file_path='output/toxic_to_benign_20240425.jsonl' --gamma=0.5 --beta=1.0 --batch_size=32 --learning_rate=1e-4 --epochs=2 --train_anti_expert
# python run_baseline_ft.py --no_save --training_file_path='output/toxic_to_benign_20240425.jsonl' --gamma=0.5 --beta=1.5 --batch_size=32 --learning_rate=1e-4 --epochs=2 --test --train_anti_expert
