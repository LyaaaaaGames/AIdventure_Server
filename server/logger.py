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
#--
#--   18/09/2023 Lyaaaaa
#--     - Added delete_log_file function.
#--     - Updated init_logger to call delete_log_file.
#---------------------------------------------------------------------------
import logging
import config
import os

log = None

def init_logger():
  global log
  delete_log_file()
  logging.basicConfig(filename = config.LOG_FILENAME,
                      filemode = config.LOG_FILEMODE,
                      format   = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                      datefmt  = '%H:%M:%S')
  log = logging.getLogger("AIdventure_Server")
  log.setLevel(config.LOG_LEVEL)
  log.addHandler(logging.StreamHandler())


#------------------------------------------------------------------------------
#
#------------------------------------------------------------------------------
def delete_log_file():
  if os.path.exists(config.LOG_FILENAME):
    os.remove(config.LOG_FILENAME)
