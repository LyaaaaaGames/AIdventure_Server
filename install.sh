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
#--  - Create the conda environment.
#--
#-- Anticipated changes (Leave empty if nothing to say):
#--  - Add a choice to install cuda support or not.
#--
#-- Changelog:
#--   21/02/2022 Lyaaaaa
#--     - Created the script.
#---------------------------------------------------------------------------

mini_conda_link="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh"
cache_folder="cache"
environment_prefix="server/miniconda/"
conda_environment_file="conda_config.yml"

function download_conda()
{
  mkdir $cache_folder
  wget $mini_conda_link -O "$cache_folder/miniconda.sh"
  chmod +x "$cache_folder/miniconda.sh"
}

function install_conda()
{
  ./"$cache_folder/miniconda.sh" -b -p $environment_prefix
}

function create_environment()
{
  conda_bin_path=$environment_prefix"bin/conda"
  ./$conda_bin_path env create --file $conda_environment_file
}

download_conda
install_conda
create_environment
