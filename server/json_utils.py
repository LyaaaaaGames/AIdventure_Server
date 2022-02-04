#------------------------------------------------------------------------------
#-- Copyright (c) 2021 Lyaaaaaaaaaaaaaaa
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Implementation Notes:
#--  - Just a few useful methods to manipulate json.
#-- Anticipated changes:
#--  -
#-- Changelog:
#--  - 01/08/2021 Lyaaaaa
#--    - Created the class by extracting the methods from server.py
#------------------------------------------------------------------------------

import json

class Json_Utils():
#------------------------------------------------------------------------------
# bytes_to_json
#------------------------------------------------------------------------------
  @staticmethod
  def bytes_to_json(p_encoded_message, p_encoding = "utf-8"):
      decoded_message = p_encoded_message.decode(p_encoding)

      json_message = json.dumps(decoded_message)
      json_message = json.loads(decoded_message)

      return json_message


#------------------------------------------------------------------------------
# json_to_bytes
#------------------------------------------------------------------------------
  @staticmethod
  def json_to_bytes(p_dict, p_encoding = "utf-8"):
    json_message = json.dumps(p__dict)
    encoded_message = json_message.encode(p_encoding)
    return encoded_message


#------------------------------------------------------------------------------
# json_to_string
#------------------------------------------------------------------------------
  @staticmethod
  def json_to_string(p_dict):
    string = json.dumps(p_dict)
    return string


