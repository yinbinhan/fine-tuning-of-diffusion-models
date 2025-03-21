# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluation of a trained model given a prompt."""

from argparse import ArgumentParser  # pylint: disable=g-importing-member
import random
from diffusers import DDIMScheduler  # pylint: disable=g-importing-member
from diffusers import StableDiffusionPipeline  # pylint: disable=g-importing-member
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer  # pylint: disable=g-multiple-import
from transformers import CLIPModel, CLIPProcessor  # pylint: disable=g-multiple-import
import functools
import ImageReward as imagereward
from pipeline_stable_diffusion_extended import StableDiffusionPipelineExtended
from scheduling_ddim_extended import DDIMSchedulerExtended
import utils
from diffusers import UNet2DConditionModel
import pandas as pd
import os


def _collect_rollout(args, pipe, is_ddp, batch, calculate_reward, state_dict):
  """Collects trajectories."""
  for _ in range(args.g_step):
    # samples for each prompt
    # collect the rollout data from the custom sampling function
    # (modified in pipeline_stable_diffusion.py and scheduling_ddim.py)
    with torch.no_grad():
      (
          image,
          latents_list,
          unconditional_prompt_embeds,
          guided_prompt_embeds,
          log_prob_list,
          _,
      ) = pipe.forward_collect_traj_ddim(prompt=batch, is_ddp=is_ddp)
      reward_list = []
      txt_emb_list = []
      for i in range(len(batch)):
        reward, txt_emb = calculate_reward(image[i], batch[i])
        reward_list.append(reward)
        txt_emb_list.append(txt_emb)
      reward_list = torch.stack(reward_list).detach().cpu()
      txt_emb_list = torch.stack(txt_emb_list).detach().cpu()
      # store the rollout data
      for i in range(len(latents_list) - 1):
        # deal with a batch of data in each step i
        state_dict["prompt"].extend(batch)
        state_dict["state"] = torch.cat((state_dict["state"], latents_list[i]))
        state_dict["next_state"] = torch.cat(
            (state_dict["next_state"], latents_list[i + 1])
        )
        state_dict["timestep"] = torch.cat(
            (state_dict["timestep"], torch.LongTensor([i] * len(batch)))
        )
        state_dict["final_reward"] = torch.cat(
            (state_dict["final_reward"], reward_list)
        )
        state_dict["unconditional_prompt_embeds"] = torch.cat((
            state_dict["unconditional_prompt_embeds"],
            unconditional_prompt_embeds,
        ))
        state_dict["guided_prompt_embeds"] = torch.cat(
            (state_dict["guided_prompt_embeds"], guided_prompt_embeds)
        )
        state_dict["txt_emb"] = torch.cat((state_dict["txt_emb"], txt_emb_list))
        state_dict["log_prob"] = torch.cat(
            (state_dict["log_prob"], log_prob_list[i])
        )
      del (
          image,
          latents_list,
          unconditional_prompt_embeds,
          guided_prompt_embeds,
          reward_list,
          txt_emb_list,
          log_prob_list,
          reward,
          txt_emb,
      )
      torch.cuda.empty_cache()

def _calculate_reward_ir(
    pipe,
    args,
    reward_tokenizer,
    tokenizer,
    weight_dtype,
    reward_clip_model,
    image_reward,
    imgs,
    prompts,
    test_flag=False,
):
  """Computes reward using ImageReward model."""
  if test_flag:
    image_pil = imgs
  else:
    image_pil = pipe.numpy_to_pil(imgs)[0]
  blip_reward, _ = utils.image_reward_get_reward(
      image_reward, image_pil, prompts, weight_dtype
  )
  if args.reward_filter == 1:
    blip_reward = torch.clamp(blip_reward, min=0)
  inputs = reward_tokenizer(
      prompts,
      max_length=tokenizer.model_max_length,
      padding="do_not_pad",
      truncation=True,
  )
  input_ids = inputs.input_ids
  padded_tokens = reward_tokenizer.pad(
      {"input_ids": input_ids}, padding=True, return_tensors="pt"
  )
  txt_emb = reward_clip_model.get_text_features(
      input_ids=padded_tokens.input_ids.to("cuda").unsqueeze(0)
  )
  return blip_reward.cpu().squeeze(0).squeeze(0), txt_emb.squeeze(0)

def _calculate_loss(args,
    state_dict,
    pipe,
    unet_copy,
    start
    ):
  for idx in range(start, start+500, args.p_batch_size):
    indices = np.arange(idx, idx+args.p_batch_size)
    with torch.no_grad():
        batch_state = state_dict["state"][indices]
        batch_next_state = state_dict["next_state"][indices]
        batch_timestep = state_dict["timestep"][indices]
        batch_final_reward = state_dict["final_reward"][indices]
        batch_unconditional_prompt_embeds = state_dict[
            "unconditional_prompt_embeds"
        ][indices]
        batch_guided_prompt_embeds = state_dict["guided_prompt_embeds"][indices]
        batch_promt_embeds = torch.cat(
            [batch_unconditional_prompt_embeds, batch_guided_prompt_embeds]
        )
        batch_txt_emb = state_dict["txt_emb"][indices]
        batch_log_prob = state_dict["log_prob"][indices]
        # calculate loss from the custom function
        # (modified in pipeline_stable_diffusion.py and scheduling_ddim.py)
        log_prob, kl_regularizer = pipe.forward_calculate_logprob(
            prompt_embeds=batch_promt_embeds.cuda(),
            latents=batch_state.cuda(),
            next_latents=batch_next_state.cuda(),
            ts=batch_timestep.cuda(),
            unet_copy=unet_copy,
            is_ddp=False,
        )
        kl = kl_regularizer.mean(dim=(1, 2, 3)).detach().cpu()
        state_dict["kl"] = torch.cat(
            (state_dict["kl"], kl)
        )
        print(batch_timestep)
    
  loss = -args.reward_weight * batch_final_reward + args.kl_weight * kl
  state_dict["loss"] = torch.cat(
            (state_dict["loss"], loss)
        )
  print("loss", loss)

def model_eval(args, model_path, step):
  torch.manual_seed(args.seed)
  torch.cuda.manual_seed(args.seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
  np.random.seed(args.seed)
  random.seed(args.seed)

  weight_dtype = torch.float16
  pipe = StableDiffusionPipelineExtended.from_pretrained(
        args.pretrained_model_name_or_path, torch_dtype=weight_dtype
    )
  pipe.scheduler = DDIMSchedulerExtended.from_config(pipe.scheduler.config)
  pipe.unet.load_attn_procs(model_path)
  pipe.unet.requires_grad_(False)
  pipe.to("cuda")
  unet_copy = UNet2DConditionModel.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="unet",
      revision=args.non_ema_revision,
  )
  unet_copy.requires_grad_(False)
  unet_copy.to("cuda", dtype=weight_dtype)
  
  
  prompt_list = [args.prompt for i in range(10)]
  
  reward_tokenizer = CLIPTokenizer.from_pretrained(
      "openai/clip-vit-large-patch14"
  )

  tokenizer = CLIPTokenizer.from_pretrained(
      args.pretrained_model_name_or_path,
      subfolder="tokenizer",
      revision=args.revision,
  )

  

  reward_clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")
  reward_clip_model.requires_grad_(False)
  reward_clip_model.to('cuda', dtype=weight_dtype)

  image_reward = imagereward.load("ImageReward-v1.0")
  image_reward.requires_grad_(False)
  image_reward.to("cuda", dtype=weight_dtype)

  calculate_reward = functools.partial(
        _calculate_reward_ir,
        pipe,
        args,
        reward_tokenizer,
        tokenizer,
        weight_dtype,
        reward_clip_model,
        image_reward,
    )
  
  def _map_cpu(x):
    return x.cpu()

  state_dict = {}
  state_dict["prompt"] = []
  state_dict["state"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["next_state"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["timestep"] = _map_cpu(torch.LongTensor())
  state_dict["final_reward"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["unconditional_prompt_embeds"] = _map_cpu(
      torch.FloatTensor().to(weight_dtype)
  )
  state_dict["guided_prompt_embeds"] = _map_cpu(
      torch.FloatTensor().to(weight_dtype)
  )
  state_dict["txt_emb"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["log_prob"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["kl"] = _map_cpu(torch.FloatTensor().to(weight_dtype))
  state_dict["loss"] = _map_cpu(torch.FloatTensor().to(weight_dtype))

  for i in range(5):
    _collect_rollout(args, pipe, False, prompt_list, calculate_reward, state_dict)
    _calculate_loss(args, state_dict, pipe, unet_copy, i*500)
  
  row_data = {
        "step": step,
        "loss_mean": state_dict['loss'].mean().item(), 
        "loss_std": state_dict['loss'].std().item(),
        "kl_mean": state_dict['kl'].mean().item(), 
        "kl_std": state_dict['kl'].std().item(), 
        "reward_mean": state_dict['final_reward'].mean().item(), 
        "reward_std": state_dict['final_reward'].std().item()
    }
  
  return row_data
  



def parse_args():
  """Parse command line flags."""
  parser = ArgumentParser()
  parser.add_argument("--prompt", default="A green colored rabbit.")
  parser.add_argument("--model-path", default="./test")
  parser.add_argument("--seed", default=0, type=int)
  parser.add_argument(
      "--pretrained_model_name_or_path",
      type=str,
      default="runwayml/stable-diffusion-v1-5",
      # required=True,
      help=(
          "Path to pretrained model or model identifier from"
          " huggingface.co/models."
      ),
  )
  parser.add_argument(
      "--revision",
      type=str,
      default=None,
      required=False,
      help=(
          "Revision of pretrained model identifier from huggingface.co/models."
      ),
  )
  parser.add_argument(
      "--g_step", type=int, default=1, help="The number of sampling steps"
  )
  parser.add_argument(
      "--reward_filter",
      type=int,
      default=0,
      help="0: raw value, 1: took positive",
  )
  parser.add_argument(
      "--non_ema_revision",
      type=str,
      default=None,
      required=False,
      help=(
          "Revision of pretrained non-ema model identifier. Must be a branch,"
          " tag or git identifier of the local or remote repository specified"
          " with --pretrained_model_name_or_path."
      ),
  )
  parser.add_argument(
      "--reward_weight", type=float, default=10, help="weight of reward loss"
  )
  parser.add_argument(
      "--kl_weight", type=float, default=0.01, help="weight of kl loss"
  )
  parser.add_argument(
      "--p_batch_size",
      type=int,
      default=10,
      help=(
          "batch size for policy update per gpu, before gradient accumulation;"
          " total batch size per gpu = gradient_accumulation_steps *"
          " p_batch_size"
      ),
  )
  parser.add_argument(
      "--num_samples",
      type=int,
      default=10,
      help="number of samples to generate per prompt",
  )
  args = parser.parse_args()
  return args

if __name__ == "__main__":
  args = parse_args()
  results = []

  for folder_name in os.listdir(args.model_path):
        subfolder_path = os.path.join(args.model_path, folder_name)

        # Check it’s a directory and starts with "save_"
        if os.path.isdir(subfolder_path) and folder_name.startswith("save_"):
            # parse out the number after "save_"
            step_str = folder_name[len("save_"):]
            try:
                step = int(step_str)
            except ValueError:
                # if it isn't a valid integer, skip
                continue
            print(f"Evaluating model in: {subfolder_path}")
#             # Step 2: Evaluate this model
            data = model_eval(args, subfolder_path, step)
#             # Step 3: Collect results
            results.append(data)
  df = pd.DataFrame(results)

  df.to_csv(args.model_path + "/model_eval.csv", index=False)
  
