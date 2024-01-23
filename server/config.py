#---------------------------------------------------------------------------
#-- Copyright (c) 2021-present LyaaaaaGames
#-- Copyright (c) 2022-present AIdventure_Server contributors
#--
#-- author : Lyaaaaa
#--
#-- Portability Issues (Leave empty if nothing to say):
#--  -
#--
#-- Implementation Notes (Leave empty if nothing to say):
#--  - This is the config file used by the server.
#--  - The settings here have priority over the client's settings.
#--      Setting them to None will give the priority to the client.
#--
#-- Anticipated changes (Leave empty if nothing to say):
#--  -
#--
#-- Changelog:
#--   21/02/2022 Lyaaaaa
#--     - Created the file.
#--
#--   24/02/2022 Lyaaaaa
#--     - Set HOST default value to 0.0.0.0
#--
#--   25/03/2022 Lyaaaaa
#--     - Set LOG_FILEMODE default value to 'a'
#--
#--   21/05/2022 Lyaaaaa
#--     - Set LOG_FILEMODE default value to 'w'
#--     - Set LOG_LEVEL default value to DEBUG (Experimental branch only)
#--
#--   09/11/2022 Lyaaaaa
#--     - Set LOG_LEVEL default value back to INFO
#--
#--   04/05/2022 Lyaaaaa
#--     - Added a new section "Models". This section contains settings for the
#--         Model class.
#--
#--   05/05/2022 Lyaaaaa
#--     - Import torch_dtype to support the usage of an enum for the dtypes.
#--     - Added OFFLOAD_DICT to the settings. When True, it avoids RAM peak when
#--         loading a model.
#--
#--   18/09/2023 Lyaaaaa
#--     - LOG_FILEMODE default value is now "a" again. The log file is now
#--         manually deleted to avoid losing logs.
#--
#--   23/01/2024 Lyaaaaa
#--     - Removed TOKENIZERS_PATH.
#---------------------------------------------------------------------------

import logging
from torch_dtype import Torch_Dtypes

# Network
HOST = "0.0.0.0"
PORT = 9999

# Logs
LOG_FILENAME = "server_logs.text"
LOG_FILEMODE = "a"
LOG_LEVEL    = logging.INFO

# Models.
#See possible values here: https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.from_pretrained

MODELS_PATH        = "models/"
DEFAULT_MODEL      = "EleutherAI/gpt-neo-125M"
ALLOW_DOWNLOAD     = None # True/False/None. If True, the server will download AI's files.
ALLOW_OFFLOAD      = None # True/False/None
OFFLOAD_FOLDER     = "offload-" # Prefix to the temp folder.
LOW_CPU_MEM_USAGE  = None # True/False/None
LIMIT_MEMORY       = None # True/False/None
OFFLOAD_DICT       = None # True/False/None

# https://huggingface.co/docs/accelerate/main/en/usage_guides/big_modeling#designing-a-device-map
# MAX_MEMORY must be a dict. E.G {0: "30GB", 1: "46GB", [x: "yMB/yGB"], "cpu": "20000MB"}. x is a gpu.
MAX_MEMORY        = None # None/dict/See documentation
DEVICE_MAP        = None # None/see documentation
TORCH_DTYPE       = None # "Auto"/None/torch.dtype/See torch_dtype.py for more info.

