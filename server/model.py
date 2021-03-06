#------------------------------------------------------------------------------
#-- Copyright (c) 2021-2022 LyaaaaaGames
#-- Copyright (c) 2022 AIdventure_Server contributors
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
#------------------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from model_type   import Model_Type
import os
import torch

# Custom imports
import logger

class Model():

  _Tokenizer  : AutoTokenizer
  _model_type : Model_Type
  _Model = None

#------------------------------------------------------------------------------
#-- __init__
#------------------------------------------------------------------------------
  def __init__(self,
               p_model_name = "EleutherAI/gpt-neo-125M",
               p_model_type = Model_Type.GENERATION.value,
               p_use_gpu    = True,):
    self._tokenizer_path   = "tokenizers/" + p_model_name
    self._model_path       = "models/" + p_model_name
    self._model_name       = p_model_name
    self.is_cuda_available = torch.cuda.is_available()
    self.is_gpu_enabled    = False
    self._model_type       = p_model_type

    if self._load() == False:
      self._download()
    if p_use_gpu == True and self.is_cuda_available == True:
      self._enable_gpu()
    else:
      logger.log.info("Model successfully loaded from local file")

#------------------------------------------------------------------------------
#-- _load
#------------------------------------------------------------------------------
  def _load(self):

    try:
      self._Tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
    except:
      logger.log.info("Token file in '" + self._tokenizer_path + "' not found.")
      return False

    try:
      if self._model_type == Model_Type.GENERATION.value:
        self._Model = AutoModelForCausalLM.from_pretrained(self._model_path)
      elif self._model_type == Model_Type.TRANSLATION.value:
        self._Model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path)

    except error:
      logger.log.error(error)
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
#-- _enable_gpu
#------------------------------------------------------------------------------
  def _enable_gpu(self):
    logger.log.info("Enabling gpu")
    self._empty_gpu_cache()
    self._get_gpu_info()

    try:
      self._Model.to("cuda")
      self.is_gpu_enabled = True
      self._get_gpu_info()

    except:
      logger.log.error("An error happened while using the GPU!")
      logger.log.info("Falling back to CPU.")
      self._disable_gpu()


#------------------------------------------------------------------------------
#-- _disable_gpu
#------------------------------------------------------------------------------
  def _disable_gpu(self):
    logger.log.info("Falling back to CPU.")
    self._Model.to("cpu")
    self._empty_gpu_cache()
    self.is_gpu_enabled = False


#------------------------------------------------------------------------------
#-- _empty_gpu_cache
#------------------------------------------------------------------------------
  def _empty_gpu_cache(self):
    logger.log.debug("Clearing GPU cache")
    torch.cuda.empty_cache()


#------------------------------------------------------------------------------
#-- _get_gpu_info
#------------------------------------------------------------------------------
  def _get_gpu_info(self):
    logger.log.debug("---------------Memory allocated---------------")
    logger.log.debug(torch.cuda.memory_allocated())
    logger.log.debug("---------------Max memory allocated---------------")
    logger.log.debug(torch.cuda.max_memory_allocated())
    logger.log.debug("---------------Memory reserved---------------")
    logger.log.debug(torch.cuda.memory_reserved())
    logger.log.debug("---------------Max memory reserved---------------")
    logger.log.debug(torch.cuda.max_memory_reserved())

