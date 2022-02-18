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
#------------------------------------------------------------------------------

from model import Model

class Translator(Model):

#------------------------------------------------------------------------------
#-- translate_text
#------------------------------------------------------------------------------
  def translate_text(self, p_input : str):
    inputs         = self._Tokenizer(p_input, return_tensors="pt")
    input_ids      = inputs.input_ids
    attention_mask = inputs.attention_mask

    outputs = self._Model.generate(input_ids,
                                   attention_mask=attention_mask,
                                   max_length=40,
                                   num_beams=4,
                                   early_stopping=True)
    generated_text = self._Tokenizer.decode(outputs[0], skip_special_tokens = True)

    return generated_text
