# Stochastic Control for Fine-tuning Diffusion Models

Code for paper: [Stochastic Control for Fine-tuning Diffusion Models: Optimality, Regularity, and Convergence](https://arxiv.org/abs/2412.18164)

This code is based on [DPOK](https://github.com/google-research/google-research/tree/master/dpok) 

## Env Installation

Follow the DPOK to install the environment.

## Training

### For a single prompt

'''
accelerate launch --num_processes=1  train_online_pg.py --max_train_steps 1000 --p_step 1 --p_batch_size 5 --v_flag 1 --g_batch_size 1 --reward_weight 10 --kl_weight 0.01 --learning_rate 3e-4 --gradient_accumulation_steps 100 --buffer_size 1000 --num_samples 10 --single_prompt "A green colored rabbit." --ratio_clip 1e-4 --dpok 0
'''

Compared to DPOK, we sample 10 trajectories for each gradient update and disable the replay buffer.

## Test

Put the lora weights under ./test, and run test.py --prompt "A green colored rabbit.". One can modify the test prompt and model path as needed in the arguments.




Updated in Aug 2025.
