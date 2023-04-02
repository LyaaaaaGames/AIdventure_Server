#------------------------------------------------------------------------------
#-- Copyright (c) 2021-present LyaaaaaGames
#-- Copyright (c) 2022-present AIdventure_Server contributors
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
#--
#--  - 01/03/2022 Lyaaaaa
#--    - Added GenerationConfig import from transformers.
#--    - Because max_length has been replaced by max_new_tokens, it is no more
#--        needed to have max_length = max_length + model_input's length.
#--    - p_parameters aren't sent into generate() anymore. They are now given
#--        to a GenerationConfig object which is an attribute (generation_config)
#--        of the Model. generate() automatically uses these config.
#------------------------------------------------------------------------------

from model import Model
from transformers import GenerationConfig

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

    self._Model.generation_config = GenerationConfig(**p_parameters)

    model_output = self._Model.generate(**model_input)
    generated_text = self._Tokenizer.decode(model_output[0], skip_special_tokens=True)

    return generated_text
