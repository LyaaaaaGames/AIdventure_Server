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
#------------------------------------------------------------------------------

from model import Model

class Translator(Model):

#------------------------------------------------------------------------------
#-- translate_text
#------------------------------------------------------------------------------
  def translate_text(self, p_input : str, p_parameters = {}):
    inputs  = self._Tokenizer([p_input], return_tensors = "pt")


    outputs = self._Model.generate(**inputs, **p_parameters)

    generated_text = self._Tokenizer.batch_decode(outputs, skip_special_tokens=True, use_cache=True)

    return generated_text[0]
