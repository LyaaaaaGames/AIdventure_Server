#------------------------------------------------------------------------------
#-- Copyright (c) 2021-2022 LyaaaaaGames
#-- Copyright (c) 2022 AIdventure_Server contributors
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Implementation Notes:
#--  -
#-- Anticipated changes:
#--  - Add more values to the enum.
#--  - Re-arrange with some logic the orders.
#-- Changelog:
#--  - 31/08/2021 Lyaaaaa
#--    - Created the file
#--
#--  - 01/09/2021 Lyaaaaa
#--    - Added LOAD_MODEL value to the enumeration.
#--
#--  - 09/09/2021 Lyaaaaa
#--    - Removed EDIT_CONFIG and updated the enums order.
#--
#--  - 28/09/2021 Lyaaaaa
#--    - Removed RELOAD_MODEL and updated the enums value.
#--
#--  - 21/12/2021 Lyaaaaa
#--    - Added LOADED_MODEL.
#--
#--  - 29/12/2021 Lyaaaaa
#--    - Added DOWNLOAD_MODEL and DOWNLOADED_MODEL requests.
#--
#--  - 01/03/2022 Lyaaaaa
#--    - Added TEXT_TRANSLATION request.
#------------------------------------------------------------------------------

from enum import Enum

class Request(Enum):
  TEXT_GENERATION  = 1
  LOAD_MODEL       = 2
  SHUTDOWN         = 3
  LOADED_MODEL     = 4
  DOWNLOAD_MODEL   = 5
  DOWNLOADED_MODEL = 6
  TEXT_TRANSLATION = 7
