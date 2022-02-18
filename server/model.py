#------------------------------------------------------------------------------
#-- Copyright (c) 2021 Lyaaaaaaaaaaaaaaa
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Portability Issues (feel free to remove this part if there is none):
#--  -
#--
#-- Implementation Notes:
#--  - This class will handle all the task related to the AI model.
#--
#-- Anticipated changes:
#--  - Update generate_text to make it possible to generate multiple answers.
#--
#-- Changelog:
#--  - 31/08/2021 Lyaaaaa
#--    - Created the file
#--
#--  - 01/09/2021 Lyaaaaa
#--    - Implemented reload_config and generate_text
#--    - Replaced the TFautoModel class by AutoModelForCausalLM.
#--
#--  - 07/09/2021 Lyaaaaa
#--    - Replaced GPT2Config by GPTNeoConfig
#--    - Replaced TFAutoModelForCausalLM for AutoModelForCausalLM (to use Pytorch).
#--
#--  - 09/09/2021 Lyaaaaa
#--    - Updated __init__ to print a text when the model is loaded.
#--    - Updated reload_config, removed the config from pretrained function.
#--    - Updated the try except in _load function.
#--    - Fixed the indents in _download.
#--
#--  - 14/09/2021 Lyaaaaa
#--    - Updated generate_text to receive a new parameter, p_parameters.
#--        The parameter is a dictionnary of parameters used by the model for
#--        the generation of text.
#--
#--  - 27/09/2021 Lyaaaaa
#--    - Updated generate_text to receive the memory and add it to the
#--        model's input.
#--    - Updated _load to remove the control of the configuration file (As there
#--        is no configuration file anymore).
#--
#--  - 04/11/2021 Lyaaaaa
#--    - Changed the default value of p_model_name in __init__ from gpt2 to
#--        gpt-neo-125M.
#--
#--  - 17/01/2022 Lyaaaaa
#--    - Added support for cuda.
#--
#--  - 18/01/2022 Lyaaaaa
#--    - Added _enable_gpu method.
#--    - Removed use_cuda parameter from from_pretrained calls as it was an error.
#--    - Updated __init__ to receive p_gpu_enabled parameter and to call _enable_gpu
#--        if p_gpu_enabled is true and is_cuda_available true.
#--
#--  - 20/01/2022 Lyaaaaa
#--    - Updated __init__ to fix the condition to call _enable_gpu and set
#--        the is_gpu_enabled attribute.
#--    - Updated generate_text to format the input_ids for cuda in case the
#--        GPU is used.
#--    - Updated _enable_gpu to set is_gpu_enabled.
#--
#--  - 21/01/2022 Lyaaaaa
#--    - Renamed p_gpu_enabled into p_use_gpu.
#--
#--  - 25/01/2022 Lyaaaaa
#--    - Added logging package and _logger attribute to debug.
#--    - Added _empty_gpu_cache and _get_gpu_info methods.
#--    - Updated generate_text and _enable_gpu to call both new methods.
#--    - Fixed _load second try, "error" not being declared.
#--
#--  - 01/02/2022 Lyaaaaa
#--    - Changed the default value of p_model_name in __init__ from gpt2 to
#--        EleutherAI/gpt-neo-125M.
#--    - Updated generate_text to append the input_ids' size to max_length in
#--        p_parameters. Also added attention_mask as parameter to generate method.
#--
#--  - 11/02/2022 Lyaaaaa
#--    - Removed _get_gpu_info call from generate_text method.
#--    - Updated _enable_gpu, _empty_gpu_cache and _get_gpu_info to fix the
#--        indentation.
#--    - Updated _enable_gpu to add a try except finally conditions to avoid
#--        to crash if the GPU runs out of memory.
#--    - Added a print in __init__ to investigate the ISSUE#2.
#--    - Replaced the prints by self._logger.info.
#--
#--  - 11/02/2022 Lyaaaaa
#--    - Updated _enable_gpu to call _model.to("cpu") if the the runtime error
#--        happens with the GPU.
#--
#--  - 18/02/2022 Lyaaaaa
#--    - Removed _Config and reload_config method (not used method)
#--    - Updated generate_text to fix the "inputs on different devices" error.
#--        attention_mark was not converted to cuda while model_input was.
#--        Moved _empty_gpu_cache into a "if self.is_gpu_enabled == True"
#--    - Extracted from _enable_gpu the code which disable the gpu. This code
#--       is now in his own method, _disable_gpu.
#---   - Updated _empty_gpu_cache and _get_gpu_info to use logger in debug mode.
#--    - Set logging level to DEBUG
#--    - Set logging to overwrite the log file.
#------------------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch
import logging


global logger
logging.basicConfig(filename = "server/server_logs.text",
                    filemode = 'w',
                    format   = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt  = '%H:%M:%S')
logger = logging.getLogger("websockets.server")
logger.setLevel(logging.DEBUG) #TODO Set to info before merge in prod
logger.addHandler(logging.StreamHandler()) # TODO Remove logger before merge in prod

class Model():
  global logger
  _Tokenizer = AutoTokenizer
  _Model     = AutoModelForCausalLM
  _logger    = logger # TODO Remove logger before merge in prod


#------------------------------------------------------------------------------
#-- __init__
#------------------------------------------------------------------------------
  def __init__(self, p_model_name = "EleutherAI/gpt-neo-125M", p_use_gpu = True):
    self._tokenizer_path   = "tokenizers/" + p_model_name
    self._model_path       = "models/" + p_model_name
    self._model_name       = p_model_name
    self.is_cuda_available = torch.cuda.is_available()
    self.is_gpu_enabled    = False

    if self._load() == False:
      self._download()
    if p_use_gpu == True and self.is_cuda_available == True:
      self._enable_gpu()
    else:
      self._logger.info("Model successfully loaded from local file")


#------------------------------------------------------------------------------
#-- generate_text
#------------------------------------------------------------------------------
  def generate_text(self,
                    p_prompt     = None,
                    p_context    = None,
                    p_memory     = None,
                    p_parameters = None):

    model_input  = p_memory + p_context + p_prompt

    model_input    = p_memory + p_context + p_prompt
    tokens         = self._Tokenizer(model_input, return_tensors = "pt")
    attention_mask = tokens.attention_mask
    model_input    = tokens.input_ids

    p_parameters["max_length"] += len(model_input[0])
    
    if self.is_gpu_enabled == True:
      model_input = model_input.to("cuda")
      attention_mask = attention_mask.to("cuda")

    model_output = self._Model.generate(input_ids       = model_input,
                                        attention_mask  = attention_mask,
                                        **p_parameters)
    generated_text = self._Tokenizer.decode(model_output[0], skip_special_tokens=True)

    if self.is_gpu_enabled == True:
      self._empty_gpu_cache()

    return generated_text


#------------------------------------------------------------------------------
#-- _load
#------------------------------------------------------------------------------
  def _load(self):

    try:
      self._Tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
    except:
      self._logger.info("Token file in '" + self._tokenizer_path + "' not found.")
      return False

    try:
      self._Model = AutoModelForCausalLM.from_pretrained(self._model_path)

    except:
      return False

    return True


#------------------------------------------------------------------------------
#-- _save
#------------------------------------------------------------------------------
  def _save(self):
    self._Tokenizer.save_pretrained(self._tokenizer_path)
    self._Model.save_pretrained(self._model_path)


#------------------------------------------------------------------------------
#-- _download
#------------------------------------------------------------------------------
  def _download(self):
    model_name = self._model_name
    self._logger.info("Trying to download the tokenizer...")
    self._Tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    cache_dir       = "cache",
                                                    resume_download = True)
    self._logger.info("Trying to download the model...")
    self._Model     = AutoModelForCausalLM.from_pretrained(model_name,
                                                           cache_dir       = "cache",
                                                           resume_download = True)
    self._save()


#------------------------------------------------------------------------------
#-- _enable_gpu
#------------------------------------------------------------------------------
  def _enable_gpu(self):
    self._logger.info("Enabling gpu")
    self._empty_gpu_cache()
    self._get_gpu_info()

    try:
      self._Model.to("cuda")
      self.is_gpu_enabled = True
      self._get_gpu_info()

    except:
      self._logger.error("An error happened while using the GPU!")
      self._logger.info("Falling back to CPU.")
      self._disable_gpu()


#------------------------------------------------------------------------------
#-- _disable_gpu
#------------------------------------------------------------------------------
  def _disable_gpu(self):
    self._Model.to("cpu")
    self._empty_gpu_cache()
    self.is_gpu_enabled = False


#------------------------------------------------------------------------------
#-- _empty_gpu_cache
#------------------------------------------------------------------------------
  def _empty_gpu_cache(self):
    self._logger.debug("Clearing GPU cache")
    torch.cuda.empty_cache()


#------------------------------------------------------------------------------
#-- _get_gpu_info
#------------------------------------------------------------------------------
  def _get_gpu_info(self):
    self._logger.debug("---------------Memory allocated---------------")
    self._logger.debug(torch.cuda.memory_allocated())
    self._logger.debug("---------------Max memory allocated---------------")
    self._logger.debug(torch.cuda.max_memory_allocated())
    self._logger.debug("---------------Memory reserved---------------")
    self._logger.debug(torch.cuda.memory_reserved())
    self._logger.debug("---------------Max memory reserved---------------")
    self._logger.debug(torch.cuda.max_memory_reserved())

