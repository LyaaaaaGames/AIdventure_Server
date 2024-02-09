#------------------------------------------------------------------------------
#-- Copyright (c) 2021-present LyaaaaaGames
#-- Copyright (c) 2022-present AIdventure_Server contributors
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Implementation Notes:
#--  - A script to quickly download and shard any model before saving them.
#-- Anticipated changes:
#--  - Add an argument to specify how much max ram for the gpu and cpu.
#-- Changelog:
#--  - 05/12/2023 Lyaaaaa
#--    - Created the file
#------------------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from accelerate import Accelerator
import tempfile
import os
import requests
import sys
import torch
import argparse

name_help = "The name of the model found on huggingface.co (sometimes with the repository as well)."
name_help += " For exemple, to download https://huggingface.co/gpt2 you should enter 'gpt2'."
name_help += " To download, https://huggingface.co/beomi/llama-2-ko-7b, you should enter 'beomi/llama-2-ko-7b'."

shard_help = "If true, the checkpoints will be splitted into smaller parts."

max_shard_size_help = "The maximum size for a checkpoint before being sharded."
max_shard_size_help += "Checkpoints shard will then be each of size lower than this size."
max_shard_size_help += "Expressed as a string, needs to be digits followed by a unit (like '5MB')"

parser = argparse.ArgumentParser(description="Process parameters.")

parser.add_argument("--model_name", required = True, type = str, help = name_help)
parser.add_argument("--shard_model",    type = bool, default = True,    help = shard_help)
parser.add_argument("--max_shard_size", type = str,  default = "200MB", help = max_shard_size_help)
parser.add_argument("--save_path",      type = str,  default = "models/customs/")

parameters     = parser.parse_args()
model_name     = parameters.model_name
accelerator    = Accelerator()
model_path     = os.path.join(parameters.save_path, model_name)
temp_folder    = None
max_shard_size = parameters.max_shard_size
shard_model    = parameters.shard_model

def create_offload_folder():
  global temp_folder
  cwd = os.getcwd()
  temp_folder = tempfile.TemporaryDirectory(prefix = "cache_",
                                            dir    = cwd)

def create_path(p_path : str):
  if not os.path.exists(p_path):
    os.makedirs(p_path)


create_offload_folder()
create_path(model_path)

args = {"low_cpu_mem_usage"  : True,
        "torch_dtype" : torch.float16,
        "offload_state_dict" : True,
        #"max_memory"         : {0:"3GB", "cpu": "12GB"}, #0 is your GPU. You might want to increase both limits
        "offload_folder"     : temp_folder.name,}

Tokenizer = AutoTokenizer.from_pretrained(model_name,
                                          cache_dir       = temp_folder.name,
                                          resume_download = True)


Model = AutoModelForCausalLM.from_pretrained(model_name,
                                            cache_dir       = temp_folder.name,
                                            resume_download = True,
                                            **args)


Tokenizer.save_pretrained(model_path)

if shard_model:
  Model.save_pretrained(
    model_path,
    max_shard_size  = max_shard_size,
    is_main_process = accelerator.is_main_process,
    save_function   = accelerator.save)
else:
  Model.save_pretrained(
    model_path,
    is_main_process = accelerator.is_main_process,
    save_function   = accelerator.save)

del Tokenizer
del Model
accelerator.free_memory()
del temp_folder
