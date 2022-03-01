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
#--  -
#--
#-- Changelog:
#--  - 18/02/2022 Lyaaaaa
#--    - Created the file.
#--
#--  - 01/03/2022 Lyaaaaa
#--    - Rewrote translate_text
#------------------------------------------------------------------------------

from model import Model

class Translator(Model):

#------------------------------------------------------------------------------
#-- translate_text
#------------------------------------------------------------------------------
  def translate_text(self, p_input : str, p_parameters = {}):
    inputs  = self._Tokenizer([p_input], return_tensors = "pt")
    outputs = self._Model.generate(**inputs, **p_parameters)

    generated_text = self._Tokenizer.batch_decode(outputs, skip_special_tokens=True)

    return generated_text
