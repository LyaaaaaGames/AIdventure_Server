#------------------------------------------------------------------------------
#-- Copyright (c) 2021-present LyaaaaaGames
#-- Copyright (c) 2022-present AIdventure_Server contributors
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Implementation Notes:
#--  - A file downloader written in Python
#-- Anticipated changes:
#--  - Add a loop to call download_file for each argument passed.
#--  - Return a specific exit code depending on the output.
#--
#-- Changelog:
#--  - 29/12/2021 Lyaaaaa
#--    - Created the file.
#--
#--  - 30/12/2021 Lyaaaaa
#--    - Replaced urllib.request by requests library.
#--
#--  - 04/02/2022 Lyaaaaa
#--    - Fixed content_lenght typo and renamed it into content_length.
#------------------------------------------------------------------------------

import requests
import sys

def humanize_size(p_size, p_suffix = "b"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(p_size) < 1024.0:
            return f"{p_size:3.1f}{unit}{p_suffix}"
        p_size /= 1024.0
    return f"{p_size:.1f}Yi{p_suffix}"


def download_file(p_url, p_path = "cache/" ,p_block_size = 65536):
  file_name    = p_url.split('/')[-1]
  file         = open(p_path + file_name, 'ba',)
  file_size_dl = 0
  percentage   = 0.0
  i            = 0

  with requests.get(p_url, stream = True) as response:
    content_length = response.headers["content-length"]
    content_length = int(content_length)

    for chunk in response.iter_content(p_block_size):
      file_size_dl += len(chunk)
      file.write(chunk)

      if i == 10:
        percentage    = round(file_size_dl * 100. / content_length, 1)
        print(file_name + ':', humanize_size(file_size_dl) , '[' + str(percentage) + '%' + ']')
        i = 0

      i += 1

    file.close()
