#---------------------------------------------------------------------------
#-- Copyright (c) 2021-2022 LyaaaaaGames
#-- Copyright (c) 2022 AIdventure_Server contributors
#--
#-- author : Lyaaaaa
#--
#-- Portability Issues (Leave empty if nothing to say):
#--  -
#--
#-- Implementation Notes (Leave empty if nothing to say):
#--  - This is the config file used by the server.
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
#--     - Set LOG_FILEMODE default valie to 'a'
#---------------------------------------------------------------------------

import logging

# Network
HOST = "0.0.0.0"
PORT = 9999

# Logs
LOG_FILENAME = "server_logs.text"
LOG_FILEMODE = "a"
LOG_LEVEL    = logging.INFO
