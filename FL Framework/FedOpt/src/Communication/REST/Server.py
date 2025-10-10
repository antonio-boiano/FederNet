#!/usr/bin/env python3
from ..model_init import ModelManager
from FedOpt.Federation.Core import Federation
import asyncio
from aiohttp import web
import json, torch

class Server:
    def __init__(self, config = None):
        if config is not None:
            self.config = config
            self.protocol_name = "REST"
            print("PROTOCOL:    REST")
            print("MODE:        SERVER")
            self.ip = self.config["ip"]
            self.port = self.config["port"]
            self.model_init = ModelManager()
            self.packet_size = self.config["server_config"]["packet_size"]
            self.allowed_num_clients = self.config["server_config"]["allowed_num_clients"]
            self.federation = Federation(self.config["server_config"]["fl_method"], base_model = self.model_init.get_base_model())
            self.client_acknowledge_dict = {}
            self.client_acknowledge_flag = {}
            self.client_index_map = {}
            self.client_count = 0
            self.client_index = 0
            self.rounds_limit = self.config["server_config"]["rounds"]
            self.sleep = self.config["sleep_time"]
            self.round = 0
            self.client_round = {}
            self.main()

        else:
            raise BaseException("[ERROR] Server Couldn't load the config")
    
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
    
    async def get_client_ip(self, request):
        client_ip = request.headers.get('X-Forwarded-For', '').split(',')[0].strip() or request.remote
        client_port = request.transport.get_extra_info('peername')[1]
        return f"{client_ip}:{client_port}"
    
    async def handle_get(self, request):
        client_adr = await self.get_client_ip(request=request)
        if "::1" in client_adr:
            client_ip = client_adr
        else:
            client_ip, port_number = client_adr.split(':')
            
        if client_ip == "127.0.0.1" or client_ip == "0.0.0.0" or client_ip == "localhost":
            client_ip = client_adr
        
        if client_ip not in self.client_index_map:
            self.client_index += 1
            self.client_index_map[client_ip] = self.client_index
            self.client_count = len(self.client_index_map)
            self.client_acknowledge_flag[client_ip] = False

        if client_ip not in self.client_round:
                self.client_round[client_ip] = 0
        
        end = not (self.client_round.get(client_ip, 0) < self.rounds_limit)

        if not self.client_acknowledge_flag[client_ip]:
            
            self.client_round[client_ip] += 1
            response_obj = {"data": self.model_init.get_base_model(),
                            "end": end,
                            "wait": False}
            if end:
                await self.remove_client(request=request)
            return web.Response(text=json.dumps(self.json_serialize(response_obj)))
        else:
            response_obj = {"data": {},
                            "end": end,
                            "wait": True}
            return web.Response(text=json.dumps(self.json_serialize(response_obj)))
    
    async def handle_post(self, request):
        msg = await request.json() 
        client_adr = await self.get_client_ip(request=request)

        if "::1" in client_adr:
            client_ip = client_adr
        else:
            client_ip, port_number = client_adr.split(':')
        
        if client_ip == "127.0.0.1" or client_ip == "0.0.0.0" or client_ip == "localhost":
            client_ip = client_adr
            
        self.client_acknowledge_dict[client_ip] = self.json_deserialize(msg["data"])
        self.client_acknowledge_flag[client_ip] = True
        end = not (self.client_round[client_ip] < self.rounds_limit)
        if not msg["end"]:
            res = self.federate()
            if end:
                await self.remove_client(request=request)
            if res:
                response_obj = {"data": self.model_init.get_base_model(),
                                "end": end,
                                "wait": False
                                }
                return web.Response(text=json.dumps(self.json_serialize(response_obj)))
            else:
                response_obj = {"data": {},
                                "end": end,
                                "wait": True
                                }
                return web.Response(text=json.dumps(self.json_serialize(response_obj)))
        else:
            response_obj = {"data": {},
                            "end": end,
                            "wait": True
                            }
            return web.Response(text=json.dumps(self.json_serialize(response_obj)))
    
    
    def reset_acknowledge(self):
        self.client_acknowledge_dict.clear()
        self.client_acknowledge_flag = self.client_acknowledge_flag.fromkeys(self.client_acknowledge_flag, False) 
        
    async def remove_client(self, request):
        client_ip = await self.get_client_ip(request=request)
        self.client_acknowledge_dict.pop(client_ip, None)
        self.client_index_map.pop(client_ip, None)
        self.client_acknowledge_flag.pop(client_ip, None)
        self.client_count -= 1
        if len(self.client_index_map) == 0:
            self.round = 0

    
    def federate(self):
        received = list(self.client_acknowledge_flag.values())
        if all([self.client_count > 0, sum(received) == self.client_count]):
            print("Federation in process")
            print(f"[INFO] Federation with client number of {self.client_count}")
            method = self.config["server_config"]["fl_method"]
            print(f"[INFO] {method}...")
            new_params = self.federation.method.apply(aggregated_dict = self.client_acknowledge_dict)
            self.model_init.set_base_model(new_params)
            self.model_init.evaluate()
            self.reset_acknowledge()
            self.round += 1
            print(f"[INFO] {method}... DONE!")
            return True
        return False
        

    async def on_startup(self, app):
        print("Server started, waiting for connections...")

    def main(self):
        try:
            app = web.Application(client_max_size=1024*1024*10)
            app.router.add_get('/', self.handle_get)
            app.router.add_post('/', self.handle_post)
            app.on_startup.append(self.on_startup)
            web.run_app(app, path=self.ip, port=self.port)
    
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down server")
        except asyncio.CancelledError:
            print("Call for Cancellation")

