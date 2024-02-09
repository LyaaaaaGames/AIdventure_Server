#------------------------------------------------------------------------------
#-- Copyright (c) 2021-present LyaaaaaGames
#-- Copyright (c) 2022-present AIdventure_Server contributors
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Implementation Notes:
#--  - The class handling the text translation.
#--
#-- Anticipated changes:
#--  - Support GPU on the translator
#--  - Make sure the inputs' length isn't longer than the max sequence length
#--      specified for the model.
#--
#-- Changelog:
#--  - 18/02/2022 Lyaaaaa
#--    - Created the file.
#--
#--  - 01/03/2022 Lyaaaaa
#--    - Rewrote translate_text
#--
#--  - 03/03/2022 Lyaaaaa
#--    - translate_text now returns a string instead of a list.
#--
#--  - 22/05/2022 Lyaaaaa
#--    - Updated translate_text to support cuda if it is enabled.
#--    - Set use_cache to true.
#--
#--  - 05/05/2023 Lyaaaaa
#--    - Removed inputs.to("cuda"). The translators won't use the gpu anymore (for now).
#--        The new system isn't supported by the MarianMT models and the generator
#--        is more in need of the GPU.
#--
#--  - 23/01/2024 Lyaaaaa
#--    - Added _load_model method (extracted from model.py. Old name "_load_translator")
#--
#--  - 25/01/2024 Lyaaaaa
#--    - Added Logger and transformers imports.
#--    - Now uses polymorphism
#--      - Added _set_parameters and _download_model methods inherited from Model.
#------------------------------------------------------------------------------

from model import Model
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
import logger

class Translator(Model):

#------------------------------------------------------------------------------
#-- translate_text
#------------------------------------------------------------------------------
  def translate_text(self, p_input : str, p_parameters = {}):
    inputs  = self._Tokenizer([p_input], return_tensors = "pt")


    outputs = self._Model.generate(**inputs, **p_parameters)

    generated_text = self._Tokenizer.batch_decode(outputs, skip_special_tokens=True, use_cache=True)

    return generated_text[0]


#------------------------------------------------------------------------------
#--
#------------------------------------------------------------------------------
  def _load_model(self):
    try:
      self._Model = AutoModelForSeq2SeqLM.from_pretrained(self._model_path)

    except Exception as e:
      logger.log.error("An unexpected error happened while loading the translator")
      logger.log.error(e)
      return False

    return True


#------------------------------------------------------------------------------
#--
#------------------------------------------------------------------------------
  def _set_parameters(self, p_parameters : dict):
    pass # Translators have no parameters for now.


#------------------------------------------------------------------------------
#--
#------------------------------------------------------------------------------
  def _download_model(self):
    logger.log.info("Trying to download the Translator...")
    self._Model = AutoModelForSeq2SeqLM.from_pretrained(self._model_name,
                                                        cache_dir       = "cache",
                                                        resume_download = True)
    super()._download_model()
