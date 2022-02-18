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
#--  - 01/02/2022 Lyaaaaa
#--    - Changed the default value of p_model_name in __init__ from gpt2 to
#--        EleutherAI/gpt-neo-125M.
#--    - Updated generate_text to append the input_ids' size to max_length in
#--        p_parameters. Also added attention_mask as parameter to generate method.
#--
#--  - 18/02/2022 Lyaaaaa
#--    - Removed Config and reload_config.
#--    - Extracted generate_text method to generator class.
#--    - Added the model_type and AutoModelForSeq2SeqLM imports.
#--    - Updated __init__ to receive the a parameter to specify the model's type.
#--    - Updated _load and _download methods to use either AutoModelForCausalLM or
#--        AutoModelForSeq2SeqLM depending of the _model_type attribute's value.
#------------------------------------------------------------------------------

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
from model_type   import Model_Type
import os

class Model():

  _Tokenizer  : AutoTokenizer
  _model_type : Model_Type
  _Model = None

#------------------------------------------------------------------------------
#-- __init__
#------------------------------------------------------------------------------
  def __init__(self,
               p_model_name = "EleutherAI/gpt-neo-125M",
               p_model_type = Model_Type.GENERATION.value):
    self._tokenizer_path = "tokenizers/" + p_model_name
    self._model_path     = "models/" + p_model_name
    self._model_name     = p_model_name
    self._model_type = p_model_type

    if self._load() == False:
      self._download()
    else:
      print("Model successfully loaded from local file")

#------------------------------------------------------------------------------
#-- _load
#--
#-- Anticipated changes:
#--  - Load the configuration file from the model folder.
#------------------------------------------------------------------------------
  def _load(self):

    try:
      self._Tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_path)
    except:
      print("Token file in '" + self._tokenizer_path + "' not found.")
      return False

    try:
      if self._model_type == Model_Type.GENERATION.value:
        self._Model = AutoModelForCausalLM.from_pretrained(self._model_path)
      elif self._model_type == Model_Type.TRANSLATION.value:
        self._Model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path)

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
    print("Trying to download the tokenizer...")
    self._Tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                    cache_dir       = "cache",
                                                    resume_download = True)
    print("Trying to download the model...")
    if self._model_type == Model_Type.GENERATION.value:
      self._Model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         cache_dir       = "cache",
                                                         resume_download = True)
    elif self._model_type == Model_Type.TRANSLATION.value:
      self._Model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                          cache_dir       = "cache",
                                                          resume_download = True)
    self._save()

