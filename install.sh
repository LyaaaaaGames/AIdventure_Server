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
#--
#--   24/02/2022 Lyaaaaa
#--     - Updated the script to give the option to install cuda or not
#--
#--   22/05/2022 Lyaaaaa
#--     - Uncommented "download_conda" and "install_conda" as it breaks the installer. Oups...
#---------------------------------------------------------------------------

mini_conda_link="https://repo.anaconda.com/miniconda/Miniconda3-py39_4.10.3-Linux-x86_64.sh"
cache_folder="cache"
environment_prefix="server/miniconda/"
conda_environment_file="conda_config"

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
  if [[ $1 = true ]]
  then
    conda_environment_file=$conda_environment_file"_cuda.yml"

  elif [[ $1 = false ]]
  then
    conda_environment_file=$conda_environment_file".yml"
  fi

  conda_bin_path=$environment_prefix"bin/conda"
  ./$conda_bin_path env create --file $conda_environment_file
}

download_conda
install_conda

# Not in a function because clear doesn't work in functions. No clue why.
loop=true
cuda=false

while $loop = true
do
  read -n 1 -p "Do you want to install CUDA's support? It will download more and increase the environment's size. [Y/N]" input
  clear
  if [[ $input = 'y' ]] || [[ $input = 'Y' ]]
  then
    loop=false
    cuda=true

  elif [[ $input = 'n' ]] || [[ $input = 'N' ]]
  then
    loop=false
    cuda=false
  fi
done

create_environment "$cuda"
