#------------------------------------------------------------------------------
#-- Copyright (c) 2021-2022 LyaaaaaGames
#-- Copyright (c) 2022 AIdventure_Server contributors
#--
#-- Author : Lyaaaaaaaaaaaaaaa
#--
#-- Portability Issues (feel free to remove this part if there is none):
#--  -
#-- Implementation Notes:
#--  - The server receives data encoded in utf-8 bytes. But the function
#--      WebSocketServerProtocol.send just needs a string and encodes by itself.
#--      See the [doc](https://websockets.readthedocs.io/en/stable/api/server.html#websockets.server.WebSocketServerProtocol.send)
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
#--  - 18/02/2022 Lyaaaaa
#--    - Added the Generator and Model_Type imports.
#--    - Renamed model into generator.
#--    - Replaced the Model class by the Generator class.
#--
#--  - 01/03/2022 Lyaaaaa
#--    - Imported translator
#--    - Added translator object
#--    - Updated handler function.
#--      - Replaced the return p_data in the if by a single one at the end.
#--      - Added global translator.
#--      - Added the possibility to load either a generation or translation model.
#--      - Added a TEXT_TRANSLATION case.
#------------------------------------------------------------------------------

import asyncio
import websockets
import logging

from json_utils import Json_Utils
from request    import Request
from generator  import Generator
from translator import Translator
from model_type import Model_Type
from downloader import download_file


HOST = "localhost"
PORT = 9999

generator  = None
translator = None

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


#------------------------------------------------------------------------------
# handle_request
#------------------------------------------------------------------------------
def handle_request(p_websocket, p_data : dict):
  global generator
  global translator

  request = p_data['request']

  if request == Request.TEXT_GENERATION.value:
    prompt     = p_data['prompt']
    context    = p_data['context']
    memory     = p_data['memory']
    parameters = p_data['parameters']

    p_data['generated_text'] = generator.generate_text(prompt,
                                                       context,
                                                       memory,
                                                       parameters)
    p_data = Json_Utils.json_to_string(p_data)


  elif request == Request.SHUTDOWN.value:
    shutdown_server()

  elif request == Request.LOAD_MODEL.value:
    if p_data["model_type"] == Model_Type.GENERATION.value:
      model_name = p_data['model_name']
      generator  = Generator(model_name, Model_Type.GENERATION.value)

    elif p_data["model_type"] == Model_Type.TRANSLATION.value:
      model_name = p_data["model_name"]
      translator = Translator(model_name, Model_Type.TRANSLATION.value)

    p_data['request'] = Request.LOADED_MODEL.value
    p_data            = Json_Utils.json_to_string(p_data)


  elif request == Request.DOWNLOAD_MODEL.value:
    url        = p_data["url"]
    save_path  = p_data["save_path"]
    download_file(url, save_path)

    p_data['request'] = Request.DOWNLOADED_MODEL.value
    p_data            = Json_Utils.json_to_string(p_data)


  elif request == Request.TEXT_TRANSLATION.value:
    prompt = p_data['prompt']
    p_data["translated_text"] = translator.translate_text(prompt)
    p_data = Json_Utils().json_to_string(p_data)


  return p_data


#------------------------------------------------------------------------------
# shutdown_server
#------------------------------------------------------------------------------
def shutdown_server():
    exit()


#------------------------------------------------------------------------------
# main
#------------------------------------------------------------------------------
async def main():
  init_logger()
  async with websockets.serve(handler, HOST, PORT):
      print("Server started ws://%s:%s" % (HOST, PORT))
      await asyncio.Future()


try:
  asyncio.run(main())

except KeyboardInterrupt:
  print("Server stopped.")




