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
#--  - A script for the useful functions.
#--
#-- Anticipated changes (Leave empty if nothing to say):
#--  - Add more useful functions.
#--
#-- Changelog:
#--   05/05/2023 Lyaaaaa
#--     - Created the file.
#---------------------------------------------------------------------------

def human_readable(p_number, p_suffix="B"):
    for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
        if abs(p_number) < 1024.0:
            return f"{p_number:3.4f} {unit}{p_suffix}"
        p_number /= 1024.0
    return f"{p_number:.1f}Yi{p_suffix}"
