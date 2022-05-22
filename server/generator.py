#------------------------------------------------------------------------------
#-- Copyright (c) 2021-2022 LyaaaaaGames
#-- Copyright (c) 2022 AIdventure_Server contributors
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Implementation Notes:
#--  - The class handling the text generation.
#--
#-- Anticipated changes:
#--  -
#--
#-- Changelog:
#--  - 18/02/2022 Lyaaaaa
#--    - Created the file.
#--
#--  - 22/05/2022 Lyaaaaa
#--    - Updated generate_text to support the gpu once again. Simplified the
#--        script by merging all the models input into a single dict "model_input".
#------------------------------------------------------------------------------

from model import Model

class Generator(Model):

#------------------------------------------------------------------------------
#-- generate_text
#------------------------------------------------------------------------------
  def generate_text(self,
                    p_prompt     = None,
                    p_context    = None,
                    p_memory     = None,
                    p_parameters = None):

    model_input    = p_memory + p_context + p_prompt
    model_input    = self._Tokenizer(model_input, return_tensors = "pt")

    if self.is_gpu_enabled:
      model_input.to("cuda")

    p_parameters["max_length"] += len(model_input[0])

    model_output = self._Model.generate(**model_input,
                                        **p_parameters)
    generated_text = self._Tokenizer.decode(model_output[0], skip_special_tokens=True)

    return generated_text
