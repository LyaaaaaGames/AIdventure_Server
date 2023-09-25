#------------------------------------------------------------------------------
#-- Copyright (c) 2021-present LyaaaaaGames
#-- Copyright (c) 2022-present AIdventure_Server contributors
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Portability Issues (Leave empty if nothing to say):
#--  - Changing the enum or dtypes might have a **BAD** result with the communication
#--      with the client!
#--
#-- Implementation Notes:
#--  - An enumeration of https://pytorch.org/docs/stable/tensor_attributes.html
#--
#-- Anticipated changes:
#--  -
#-- Changelog:
#--  - 05/05/2023 Lyaaaaa
#--    - Created the file
#------------------------------------------------------------------------------

from enum import Enum
import torch

class Torch_Dtypes(Enum):
  FLOAT_32    = 0
  FLOAT_64    = 1
  COMPLEX_64  = 2
  COMPLEX_128 = 3
  FLOAT_16    = 4
  BFLOAT_16   = 5
  UINT_8      = 6
  INT_8       = 7
  INT_16      = 8
  INT_32      = 9
  INT_64      = 10
  BOOL        = 11
  AUTO        = 12

  dtypes = [torch.float32,
            torch.float64,
            torch.complex64,
            torch.complex128,
            torch.float16,
            torch.bfloat16,
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
            "auto"]
