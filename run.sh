#!/bin/bash
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
#--  - Run the server.
#--
#-- Anticipated changes (Leave empty if nothing to say):
#--  -
#--
#-- Changelog:
#--   21/02/2022 Lyaaaaa
#--     - Created the script.
#--
#--   22/02/2022 Lyaaaaa
#--     - Fix the paths being broken.
#---------------------------------------------------------------------------

environment_prefix="server/miniconda/envs/aidventure/"
python_bin="bin/python"

./$environment_prefix$python_bin server/server.py
