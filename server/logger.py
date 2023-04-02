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
#--  - This is the file handling logging.
#--
#-- Anticipated changes (Leave empty if nothing to say):
#--  -
#--
#-- Changelog:
#--   24/02/2022 Lyaaaaa
#--     - Created the file.
#---------------------------------------------------------------------------
import logging
import config

log = None

def init_logger():
  global log
  logging.basicConfig(filename = config.LOG_FILENAME,
                      filemode = config.LOG_FILEMODE,
                      format   = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                      datefmt  = '%H:%M:%S')
  log = logging.getLogger("AIdventure_Server")
  log.setLevel(config.LOG_LEVEL)
  log.addHandler(logging.StreamHandler())
