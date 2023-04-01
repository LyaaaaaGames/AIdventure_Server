# Style guide

Let's keep this project consistent.
Please follow the rules.

## File Header

Each python script must start with this header.

```
#---------------------------------------------------------------------------
#-- Copyright (c) 2021-present LyaaaaaGames
#-- Copyright (c) 2022-present AIdventure_Server contributors
#--
#-- author :
#--
#-- Portability Issues (Leave empty if nothing to say):
#--  -
#--
#-- Implementation Notes (Leave empty if nothing to say):
#--  -
#--
#-- Anticipated changes (Leave empty if nothing to say):
#--  - 
#--
#-- Changelog (Changelog has to be updated! See GPL-3.0 license):
#--   dd/mm/yyyy author
#--     -
#---------------------------------------------------------------------------
```
**After each update of a file, please update the file changelog (in the header)!**

## Coding guidelines

- Indentation     : 2 spaces (convert tabs into spaces)
- Max line length : 80 characters (As long as it doesn't make it unreadable)

### How to name things

- Variables : variable_name
- Parameters : p_parameter_name
- Classes    : Class_Name
- Objects    : Object_Name
- functions  : function_name

### Comments

```
# This is a comment.
#print("This is disabled code")
```

Disabled code is fine while testing but isn't in pull requests or production.

### Align things

Try to group and align things together (when they are related) for better readability.

```
variable1          = "something"
long_variable_name = "something else"
third_variable     = 55

logging.basicConfig(filename = "server/server_logs.text",
                    filemode = 'a',
                    format   = '%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt  = '%H:%M:%S')
```
















