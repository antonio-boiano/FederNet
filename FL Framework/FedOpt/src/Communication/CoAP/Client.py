#!/usr/bin/env python3
from ..model_init import ModelManager
import asyncio
from aiocoap import Context, Message, GET, POST
import pickle

class Client:
    def __init__(self, config=None):
        self.config = config
        self.ip = self.config["ip"]
        self.port = self.config["port"]
        self.address = f"coap://{self.ip}:{self.port}/index"
        self.model_init = ModelManager()
        self.index = 0
        self.end = False
        self.wait = False
        self.sleep = self.config["sleep_time"]
        asyncio.run(self.main())


    async def make_get_request(self, context, url):
        get_request = Message(code=GET)
        get_request.set_request_uri(url)
        try:
            response = await context.request(get_request).response
            response_data = pickle.loads(response.payload)
            self.end = response_data["end"]
            self.wait = response_data["wait"]
            if not self.wait:        
                self.model_init.set_base_model(response_data["data"])
            else:
                await asyncio.sleep(self.sleep)
        
        except Exception as e:
            self.sleep


    async def make_post_request(self, context, url):
        msg = {"data": self.model_init.get_base_model(),
               "end": self.end,
               "wait": False
               }
        data_to_send = msg
        request = Message(code=POST, uri=url, payload=pickle.dumps(data_to_send))
        await context.request(request).response

    async def main(self):
        try:
            context = await Context.create_client_context()
            while True:
                await self.make_get_request(context, self.address)
                if self.end:
                    print("End of Connection")
                    break
                if not self.wait:
                    self.model_init.train(self.config["client_config"]["rounds"], self.index)
                    await self.make_post_request(context, self.address)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down client")
