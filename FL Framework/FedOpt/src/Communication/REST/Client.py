#!/usr/bin/env python3
from ..model_init import ModelManager
import asyncio
import aiohttp
import json
import torch 

class Client:
    def __init__(self, config = None):
        self.config = config
        self.ip = self.config["ip"]
        self.port = self.config["port"]
        self.address = f"http://{self.ip}:{self.port}"
        self.model_init = ModelManager()
        self.index = 0
        self.end = False
        self.wait = False
        self.sleep = self.config["sleep_time"]
        asyncio.run(self.main())

    def json_serialize(self,msg):
        for key in msg.keys():
            if isinstance(msg[key], dict):
                msg[key] = self.json_serialize(msg[key])
            elif isinstance(msg[key], torch.Tensor):
                msg[key] = msg[key].tolist()
        return msg
    
    def json_deserialize(self,msg):
        for key in msg.keys():
            if isinstance(msg[key], dict):
                msg[key] = self.json_deserialize(msg[key])
            else:
                msg[key] = torch.tensor(msg[key])
        return msg
    
    async def make_get_request(self,session, url):
        async with session.get(url) as response:
            response_text = await response.text()
            response = json.loads(response_text)
            self.end = response["end"]
            self.wait = response["wait"]

            if not self.wait:
                des = self.json_deserialize(response["data"])
                self.model_init.set_base_model(des)
            else:
                await asyncio.sleep(self.sleep)


    async def make_post_request(self, session, url): # client send
        msg = {"data": self.model_init.get_base_model(),
                "end": self.end,
                "wait": False
                }
        
        data_to_send = self.json_serialize(msg)
        async with session.post(url, json=data_to_send) as response:
            response_text = await response.text()

       
    async def main(self):
        try:
            async with aiohttp.ClientSession() as session:
                while True: 
                    await self.make_get_request(session, self.address)
                    if self.end:
                        print("End of Connection")
                        break
                    if not self.wait:
                        self.model_init.train(self.config["client_config"]["rounds"], self.index)
                        await self.make_post_request(session, self.address)
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down client")
