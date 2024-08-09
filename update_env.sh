#!/bin/bash
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
#--  - Script to update the environment. It's better than re-installing everything.
#--
#-- Anticipated changes (Leave empty if nothing to say):
#--  -
#--
#-- Changelog:
#--   09/08/2024 Lyaaaaa
#--     - Created the script.
#---------------------------------------------------------------------------

loop=true
while $loop = true
do
  read -n 1 -p "Do you want to update with CUDA's support? It will download more and increase the environment's size. [Y/N]" input
  clear
  if [[ $input = 'y' ]] || [[ $input = 'Y' ]]
  then
    ./server/miniconda/bin/conda env update --file conda_config_cuda.yml
    loop=false

  elif [[ $input = 'n' ]] || [[ $input = 'N' ]]
  then
    ./server/miniconda/bin/conda env update --file conda_config.yml
    loop=false
  fi
done