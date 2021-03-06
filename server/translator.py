#------------------------------------------------------------------------------
#-- Copyright (c) 2021-2022 LyaaaaaGames
#-- Copyright (c) 2022 AIdventure_Server contributors
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
#------------------------------------------------------------------------------

from model import Model

class Translator(Model):

#------------------------------------------------------------------------------
#-- translate_text
#------------------------------------------------------------------------------
  def translate_text(self, p_input : str, p_parameters = {}):
    inputs  = self._Tokenizer([p_input], return_tensors = "pt")

    if self.is_gpu_enabled:
      inputs.to("cuda")

    outputs = self._Model.generate(**inputs, **p_parameters)

    generated_text = self._Tokenizer.batch_decode(outputs, skip_special_tokens=True, use_cache=True)

    return generated_text[0]
