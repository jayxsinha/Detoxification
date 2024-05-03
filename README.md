# Controlled Text Generation Project

Our project addresses the challenge of controlled text generation in large language models, which can produce unwanted or biased content. Guided generation with smaller version of larger models such as GPT2-XL and GPT2-Base with GPT2-Base being a reward model [1] has recently shown great capability in generating controlled text. 

Guided generation approaches only fine-tune smaller models and keep the larger model parameters frozen. However, finetuning the smaller model in isolation without taking the larger model parameters into account can result in text degeneration in longer sequences. We intend to measure this gap and propose to meta-train a GPT2-Base model as a reward model using RL algorithms like PPO [2], DPO [3] etc. by using larger model logits.

For a given larger model, which we call Large, we will have next token xt for given prompt x0:t−1 as:
xt = arg max z PLarge(z|x0:t)

For a fine-tuned expert model, which we call Expert, we will have next token xt for given prompt x0:t−1 as:
xt = arg max z PExpert(z|x0:t)

We intend to make the generation as the following with β as a hyper-parameter:
xt = arg max z PLarge(z|x0:t)+βPExpert(z|x0:t)

Additionally, we can use a GPT2-Base as an antiexpert termed Base:
xt = arg max z PBase(z|x0:t)

Now, the final formulation will become with α as a hyper-parameter:
xt = arg max z PLarge(z|x0:t) + βPExpert(z|x0:t) − αPBase(z|x0:t)

We will meta-train the expert model with the test time generation algorithm as outlined in equations 1 and 2 using PPO [2] and DPO [3]. We will need to run a hyperparameter search for α and β. Moreover, we will also change the formulations of the above equation to test which approach works. We will be using GPT2-Large as our Large model and GPT2-Base for the Expert and Base models.

We use TRL Library with Transformers Library to do the training.

## References
[1] [Reward Augmented Decoding](https://arxiv.org/abs/2310.09520)

[2] [PPO](https://github.com/huggingface/trl/blob/main/examples/research_projects/toxicity/scripts/gpt-j-6b-toxicity.py)

[3] [DPO](https://github.com/huggingface/trl/blob/main/examples/scripts/dpo.py)