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
#--
#--  - 05/05/2023 Lyaaaaa
#--    - The condition for moving the inputs to the gpu is now "is_cuda_available"
#--        and not checking the is_gpu_enabled attribute anymore.
#--    - Import logger to display a log when loading the inputs in the gpu.
#--    - Called _empty_gpu_cache after the generation. This releases some memory.
#--
#--  - 25/01/2024 Lyaaaaa
#--    - The class now uses polymorphism.
#--    - Added torch_dtype and more transformers imports.
#--    - Added _set_parameters and _download_model methods inherited from Model.
#--
#--  - 31/01/2024 Lyaaaaa
#--    - generate_text now longer receives memory and context as parameters.
#--        They are embedded in the prompt parameter by the client.
#------------------------------------------------------------------------------

from model        import Model
from torch_dtype  import Torch_Dtypes
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer, GenerationConfig
import logger

class Generator(Model):

#------------------------------------------------------------------------------
#-- generate_text
#------------------------------------------------------------------------------
  def generate_text(self,
                    p_prompt     = None,
                    p_parameters = None):

    model_input    = self._Tokenizer(p_prompt, return_tensors = "pt")

    if self.is_cuda_available:
      logger.log.info("Loading inputs to GPU")
      model_input.to("cuda")

    self._Model.generation_config = GenerationConfig(**p_parameters)

    model_output = self._Model.generate(**model_input)
    generated_text = self._Tokenizer.decode(model_output[0], skip_special_tokens=True)

    self._empty_gpu_cache()
    return generated_text


#------------------------------------------------------------------------------
#--
#------------------------------------------------------------------------------
  def _set_parameters(self, p_parameters : dict):
    logger.log.info("Setting up the Generator.")
    super()._set_parameters(p_parameters)

    if self._limit_memory == False:
      self._max_memory = None
    elif self._limit_memory == None and p_parameters["limit_memory"] == True:
      self._max_memory = {0     : p_parameters["max_memory"]["0"],
                          "cpu" : p_parameters["max_memory"]["cpu"]}

    if self._allow_offload == True:
      self.create_offload_folder()
    elif self._allow_offload == None and p_parameters["allow_offload"] == True:
      self.create_offload_folder()


    if self._allow_download == None:
      self._allow_download = p_parameters["allow_download"]

    if self._device_map == None:
      self._device_map = p_parameters["device_map"]

    if self._torch_dtype == None:
      self._torch_dtype = Torch_Dtypes.dtypes.value[p_parameters["torch_dtype"]]

    if self._offload_dict == None:
      self._offload_dict = p_parameters["offload_dict"]

    if self._low_memory_mode == None:
      self._low_memory_mode  = p_parameters["low_memory_mode"]


#------------------------------------------------------------------------------
#--
#------------------------------------------------------------------------------
  def _download_model(self):
    self._Model = AutoModelForCausalLM.from_pretrained(self._model_name,
                                                       cache_dir       = "cache",
                                                       resume_download = True)
    super()._download_model()

