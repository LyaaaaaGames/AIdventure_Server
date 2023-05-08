#------------------------------------------------------------------------------
#-- Copyright (c) 2021-present LyaaaaaGames
#-- Copyright (c) 2022-present AIdventure_Server contributors
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Portability Issues (feel free to remove this part if there is none):
#--  -
#-- Implementation Notes:
#--  - The server receives data encoded in utf-8 bytes. But the function
#--      WebSocketServerProtocol.send just needs a string and encodes by itself.
#--      See the [doc](https://websockets.readthedocs.io/en/stable/api/server.html#websockets.server.WebSocketServerProtocol.send)
#--  - The server uses two models for the translation instead of one handling
#--      multiple languages. It trades performances for a better translation.
#-- Anticipated changes:
#--  - Make shutdown_server function cleaner.
#--  - In handler no longer shutdown the server when an user disconnect (because
#--      once the multiplayer is implemented it will create troubles).
#--
#-- Changelog:
#--  - 27/08/2021 Lyaaaaa
#--    - Created the file
#--
#--  - 31/08/2021 Lyaaaaa
#--    - Added init_logger, handler, decode_bytes_into_json functions,
#--        shutdown_server, handle_request.
#--    - Removed the Hello function.
#--
#--  - 01/09/2021 Lyaaaaa
#--    - Extracted the json related functions to its own class and file.
#--    - Updated the calls of the json functions.
#--    - Added a "model" gloabal variable.
#--    - Updated handler function to store handle_request returning value and
#--        send it back to the client if it's not null.
#--    - Partly implemented handle_request. It now takes care of loading the
#--        model and generating text.
#--    - Corrected a syntax error in handle_request.
#--
#--  - 14/09/2021 Lyaaaaa
#--    - Updated handle_request.
#--      - Updated the TEXT_GENERATION case to receive the generation parameters
#--          from the client and give it to model.generate_text()
#--      - Updated the RELOAD_MODEL case to no longer do anything for now.
#--          Reloading the model for now isn't needed anymore.
#--
#--  - 27/09/2021 Lyaaaaa
#--    - Updated handle_request to send the memory to model.generate_text.
#--
#--  - 28/09/2021 Lyaaaaa
#--    - Created a main function and called it.
#--    - Slightly changed the main loop. It now calls asyncio.run instead of
#--        asyncio.get_event_loop.
#--    - Implemented in a dirty way shutdown server.
#--    - Updated the handler function. It no longer sends a greeting to the client
#--        and it now displays on the server the ip of the client. It also now
#--        handle exceptions when the client disconnect and calls shutdown_server
#--        on the disconnection.
#--
#--  - 21/12/2021 Lyaaaaa
#--    - Updated handle_request to send an answer to the server after the model
#--        is loaded.
#--
#--  - 29/12/2021 Lyaaaaa
#--    - Imported download_file function from downloader
#--    - Updated handle_request to handle Request.DOWNLOAD_MODEL case.
#--
#--  - 21/05/2022 Lyaaaaa
#--    - Updated handle_request to add more debug messages and to use the
#--        use_gpu value for both the generator and translator.
#--
#--  - 15/08/2022 Lyaaaaa
#--    - Updated a final except in handler. On unexpected error, the server will
#--        exit.
#--    - Updated handle_request to receive low_memory_mode value from the client.
#--    - Updated the call of Generator constructor to send it low_memory_mode
#--
#--  - 04/05/2022 Lyaaaaa
#--    - Updated handle_request to receive the new parameters and send them inside
#--        a list to the Generator class. Removed use_gpu var.
#--
#--  - 05/05/2022 Lyaaaaa
#--    - del generator now happens only when loading a new generator.
#--    - Added offload_dict to the parameters list.
#--    - Removed use_gpu from the parameters passed to the translators init.
#--    - Removed the log about the translator using the gpu.
#--
#--  - 07/05/2022 Lyaaaaa
#--    - Removed the import of the downloader and its use inside this script.
#--
#--  - 08/05/2022 Lyaaaaa
#--    - Updated handle_request to receive the limit_memory parameter from the
#--        client.
#------------------------------------------------------------------------------

import asyncio
import websockets

# Custom imports
import config
import logger

from json_utils import Json_Utils
from request    import Request
from generator  import Generator
from translator import Translator
from model_type import Model_Type


HOST = config.HOST
PORT = config.PORT

generator           = None
from_eng_translator = None
to_eng_translator   = None

#------------------------------------------------------------------------------
# init_logger
#------------------------------------------------------------------------------
def init_logger():
  logger = logging.getLogger("websockets.server")
  logger.setLevel(logging.ERROR)
  logger.addHandler(logging.StreamHandler())


#------------------------------------------------------------------------------
# handler
#------------------------------------------------------------------------------
async def handler(p_websocket, path):
  client_ip = p_websocket.remote_address[0]
  print("Client " + client_ip + " connected")

  try:
    async for message in p_websocket:
      json_message = Json_Utils.bytes_to_json(message)
      data_to_send = handle_request(p_websocket, json_message)

      if data_to_send != None:
        await p_websocket.send(data_to_send)

  except websockets.exceptions.ConnectionClosedOK:
    print("Closing the server")
    shutdown_server()

  except websockets.exceptions.ConnectionClosedError:
    print("Closing the server")
    shutdown_server()

  except:
    print("Unexpected error shutting down the server")
    shutdown_server()


#------------------------------------------------------------------------------
# handle_request
#------------------------------------------------------------------------------
def handle_request(p_websocket, p_data : dict):
  global generator
  global from_eng_translator
  global to_eng_translator

  request = p_data['request']

  if request == Request.TEXT_GENERATION.value:
    prompt     = p_data['prompt']
    context    = p_data['context']
    memory     = p_data['memory']
    parameters = p_data['parameters']

    generated_text = generator.generate_text(prompt,
                                             context,
                                             memory,
                                             parameters)

    p_data["generated_text"] = generated_text
    p_data = Json_Utils.json_to_string(p_data)

    return p_data

  elif request == Request.SHUTDOWN.value:
    shutdown_server()

  elif request == Request.LOAD_MODEL.value:
    parameters = {"low_memory_mode" : p_data['low_memory_mode'],
                  "allow_offload"   : p_data['allow_offload'],
                  "limit_memory"    : p_data['limit_memory'],
                  "max_memory"      : p_data['max_memory'],
                  "allow_download"  : p_data['allow_download'],
                  "device_map"      : p_data['device_map'],
                  "torch_dtype"     : p_data['torch_dtype'],
                  "offload_dict"    : p_data['offload_dict'],}

    if p_data["model_type"] == Model_Type.GENERATION.value:
      del generator
      logger.log.debug("loading generator")
      model_name = p_data['model_name']

      generator = Generator(model_name,
                             Model_Type.GENERATION.value,
                             parameters)
      logger.log.info("Is CUDA available: " + format(generator.is_cuda_available))

    elif p_data["model_type"] == Model_Type.TRANSLATION.value:
      logger.log.debug("loading translator")
      model_name = p_data["to_eng_model"]
      to_eng_translator = Translator(model_name,
                                     Model_Type.TRANSLATION.value,
                                     parameters)

      model_name = p_data["from_eng_model"]
      from_eng_translator = Translator(model_name,
                                       Model_Type.TRANSLATION.value,
                                       parameters)

    p_data['request'] = Request.LOADED_MODEL.value
    p_data            = Json_Utils.json_to_string(p_data)

    return p_data


  elif request == Request.TEXT_TRANSLATION.value:
    prompt = p_data["prompt"]
    to_eng = p_data["to_eng"]

    p_data["translated_text"] = translate_text(prompt, to_eng)
    p_data = Json_Utils().json_to_string(p_data)

  return p_data


#------------------------------------------------------------------------------
# translate
#------------------------------------------------------------------------------
def translate_text(p_prompt : str, p_to_eng : bool = True):
  global from_eng_translator
  global to_eng_translator

  translated_text = None
  if p_prompt == "":
    translated_text = ""
  else:
    if p_to_eng == True:
      translated_text = to_eng_translator.translate_text(p_prompt)
    else:
      translated_text = from_eng_translator.translate_text(p_prompt)

  return translated_text


#------------------------------------------------------------------------------
# shutdown_server
#------------------------------------------------------------------------------
def shutdown_server():
  exit()


#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------
async def main():
  logger.init_logger()
  async with websockets.serve(handler, HOST, PORT):
      print("Server started ws://%s:%s" % (HOST, PORT))
      await asyncio.Future()


try:
  asyncio.run(main())

except KeyboardInterrupt:
  print("Server stopped.")





