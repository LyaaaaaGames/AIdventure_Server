#------------------------------------------------------------------------------
#-- Copyright (c) 2021-present LyaaaaaGames
#-- Copyright (c) 2022-present AIdventure_Server contributors
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Portability Issues (feel free to remove this part if there is none):
#--  - This class needs Pytorch installed.
#--
#-- Implementation Notes:
#--  - This class will handle all the task related to the AI model.
#--
#-- Anticipated changes:
#--  - Use a specific configuration.json file for each model (each model folder
#--      already has a config file. It could be more efficient to save each config
#--      into its own folder.
#--  - Update generate_text to make it possible to generate multiple answers.
#--  - Get rid of the configuration as I confused the generation and model config.
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
#--  - 18/02/2022 Lyaaaaa
#--    - Extracted generate_text method to generator class.
#--    - Added the model_type and AutoModelForSeq2SeqLM imports.
#--    - Updated __init__ to receive the a parameter to specify the model's type.
#--    - Updated _load and _download methods to use either AutoModelForCausalLM or
#--        AutoModelForSeq2SeqLM depending of the _model_type attribute's value.
#--    - Removed _Config and reload_config method (not used method)
#--    - Updated generate_text to fix the "inputs on different devices" error.
#--        attention_mark was not converted to cuda while model_input was.
#--        Moved _empty_gpu_cache into a "if self.is_gpu_enabled == True"
#--    - Extracted from _enable_gpu the code which disable the gpu. This code
#--       is now in his own method, _disable_gpu.
#---   - Updated _empty_gpu_cache and _get_gpu_info to use logger in debug mode.
#--    - Set logging level to DEBUG
#--    - Set logging to overwrite the log file.
#--
#--  - 21/02/2022 Lyaaaaa
#--    - Update generate_text to add a fallback during the generation with the GPU.
#--    - Added logging message in generate_text and _disable_gpu.
#--    - Updated generate_text to catch the error in the except.
#--
#--  - 24/02/2022 Lyaaaaa
#--    - Replaced the init of logging by the import of the new script logger.
#--    - Replaced self._logger by logger.log.
#--
#--  - 15/08/2022 Lyaaaaa
#--    - Updated __init__ to receive the p_low_memory_mode parameter.
#--    - Updated _load to enable low_cpu_mem_usage option while loading the
#--        generator model.
#--    - Updated _load to fix the except being wrong.
#--    - Extracted a log print from _enable_gpu to _disable_gpu
#--    - Updated _empty_gpu_cache to torch.no_grad() otherwise the memory stays
#--        in use. Even with this solution a few hundreds of MB stays in use...
#--
#--  - 01/04/2023 Lyaaaaa
#--    - Updated __init__ to declare attribute generation_config.
#--
#--  - 04/05/2023 Lyaaaaa
#--    - Added imports. Accelerator and tempfile.
#--    - _tokenizer_path becomes _tokenizers_path.
#--    - Added create_offload_folder method. It creates a temp folder.
#--    - Added _set_model_parameters method.
#--    - Updated _empty_gpu_cache to call accelerator.free_memory()
#--    - Deleted _disable/enable_gpu methods as they aren't used anymore.
#--    - Added many attributes related to the config.
#--    - Extracted _tokenizers_path and _model_path initialization in the config.
#--    - Updated __init__
#--        Now receives the many new parameters in a list.
#--        Deleted is_gpu_enabled.
#--        Call _empty_gpu_cache before loading the model.
#--        Call _set_model_parameters before loading the model.
#--    - Added a few log message (info + debug) in __init__ and _load.
#--    - Updated _load to use the new attributes:
#--        device_map, torch_dtype, max_memory and offload_folder.
#--
#--  - 05/05/2023 Lyaaaaa
#--    - Added import of human_readable.
#--    - Updated _get_gpu_info to use the human_readable.
#--    - Updated __init__ to call _get_gpu_info after loading the model.
#------------------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from model_type   import Model_Type
from accelerate   import Accelerator

import tempfile
import os
import torch

import logger
import config
from utils import human_readable


class Model():

  _Tokenizer  : AutoTokenizer
  _model_type : Model_Type
  _Model = None

  _tokenizers_path = config.TOKENIZERS_PATH
  _model_path      = config.MODELS_PATH
  _allow_offload   = config.ALLOW_OFFLOAD
  _limit_memory    = config.LIMIT_MEMORY
  _max_memory      = config.MAX_MEMORY
  _allow_download  = config.ALLOW_DOWNLOAD
  _device_map      = config.DEVICE_MAP
  _torch_dtype     = config.TORCH_DTYPE
  _low_memory_mode = config.LOW_CPU_MEM_USAGE
  _offload_folder  = None

#------------------------------------------------------------------------------
#-- __init__
#------------------------------------------------------------------------------
  def __init__(self,
               p_model_name = config.DEFAULT_MODEL,
               p_model_type = Model_Type.GENERATION.value,
               p_parameters = {}):

    self._tokenizers_path  += p_model_name
    self._model_path       += p_model_name
    self._model_name       = p_model_name
    self.is_cuda_available = torch.cuda.is_available()
    self._model_type       = p_model_type

    self._set_model_parameters(p_parameters)

    self._empty_gpu_cache()
    if self._load() == False:
      if self._allow_download == True:
        self._download()
      else:
        logger.log.info("Couldn't load the model files.")
        logger.log.info("Downloading the model with the server is disabled.")
    else:
      logger.log.info("Model successfully loaded from local file")
      self._get_gpu_info()


#------------------------------------------------------------------------------
#-- _set_model_parameters
#------------------------------------------------------------------------------
  def _set_model_parameters(self, p_parameters : dict):
    logger.log.info("Setting up the model.")
    logger.log.debug(p_parameters)

    if self._low_memory_mode == None:
      self._low_memory_mode  = p_parameters["low_memory_mode"]

    if self._limit_memory == False:
      self._max_memory = None
    elif self._limit_memory == None:
      self._max_memory = p_parameters["max_memory"]

    if self._allow_offload == True:
      create_offload_folder()
    elif self._allow_offload == None and p_parameters["allow_offload"] == True:
      create_offload_folder()


    if self._allow_download == None:
      self._allow_download = p_parameters["allow_download"]

    if self._device_map == None:
      self._device_map = p_parameters["device_map"]

    if self._torch_dtype == None:
      self._torch_dtype = p_parameters["torch_dtype"]

#------------------------------------------------------------------------------
#-- _load
#------------------------------------------------------------------------------
  def _load(self):

    try:
      self._Tokenizer = AutoTokenizer.from_pretrained(self._tokenizers_path)
    except Exception as e:
      logger.log.info("Token file in '" + self._tokenizers_path + "' not found.")
      logger.log
      return False

    try:
      if self._model_type == Model_Type.GENERATION.value:
        args = {"low_cpu_mem_usage": self._low_memory_mode,
                "device_map"       : self._device_map,
                "torch_dtype"      : self._torch_dtype,
                "max_memory"       : self._max_memory,
                "offload_folder"   : self._offload_folder}

        logger.log.debug("Model settings:")
        logger.log.debug(args)
        self._Model = AutoModelForCausalLM.from_pretrained(self._model_path,
                                                           **args)

      elif self._model_type == Model_Type.TRANSLATION.value:
        self._Model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path)

    except Exception as e:
      logger.log.error("An unexpected error happened while loading the model: ")
      logger.log.error(e)
      return False

    return True


#------------------------------------------------------------------------------
#-- _save
#------------------------------------------------------------------------------
  def _save(self):
    self._Tokenizer.save_pretrained(self._tokenizers_path)
    self._Model.save_pretrained(self._model_path)


#------------------------------------------------------------------------------
#-- _download
#------------------------------------------------------------------------------
  def _download(self):
    model_name = self._model_name
    logger.log.info("Trying to download the tokenizer...")
    self._Tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    cache_dir       = "cache",
                                                    resume_download = True)
    logger.log.info("Trying to download the model...")
    if self._model_type == Model_Type.GENERATION.value:
      self._Model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         cache_dir       = "cache",
                                                         resume_download = True)
    elif self._model_type == Model_Type.TRANSLATION.value:
      self._Model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                          cache_dir       = "cache",
                                                          resume_download = True)
    self._save()

#------------------------------------------------------------------------------
#-- _empty_gpu_cache
#------------------------------------------------------------------------------
  def _empty_gpu_cache(self):
    logger.log.debug("Clearing GPU cache")
    accelerator = Accelerator()
    accelerator.free_memory()

    with torch.no_grad():
      torch.cuda.empty_cache()
    self._get_gpu_info()

#------------------------------------------------------------------------------
#-- _get_gpu_info
#------------------------------------------------------------------------------
  def _get_gpu_info(self):
    logger.log.debug("---------------Memory allocated---------------")
    logger.log.debug(human_readable(torch.cuda.memory_allocated()))
    logger.log.debug("---------------Max memory allocated---------------")
    logger.log.debug(human_readable(torch.cuda.max_memory_allocated()))
    logger.log.debug("---------------Memory reserved---------------")
    logger.log.debug(human_readable(torch.cuda.memory_reserved()))
    logger.log.debug("---------------Max memory reserved---------------")
    logger.log.debug(human_readable(torch.cuda.max_memory_reserved()))

  #------------------------------------------------------------------------------
  # create_offload_folder
  #------------------------------------------------------------------------------
  def create_offload_folder():
      cwd = os.getcwd()
      folder = tempfile.TemporaryDirectory(prefix = config.OFFLOAD_FOLDER,
                                           dir    = cwd)
      config.OFFLOAD_FOLDER = folder
      self._offload_folder  = config.OFFLOAD_FOLDER.name
