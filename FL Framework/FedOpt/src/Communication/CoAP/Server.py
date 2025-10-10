#!/usr/bin/env python3
from ..model_init import ModelManager
from FedOpt.Federation.Core import Federation
import asyncio
from aiocoap import Context, Message, GET, POST
from aiocoap.resource import Resource, Site
import pickle

class Server:
    def __init__(self, config=None):
        if config is not None:
            self.config = config
            self.protocol_name = "CoAP"
            print("PROTOCOL:    CoAP")
            print("MODE:        SERVER")
            self.ip = self.config["ip"]
            self.port = self.config["port"]
            self.model_init = ModelManager()
            self.packet_size = self.config["server_config"]["packet_size"]
            self.allowed_num_clients = self.config["server_config"]["allowed_num_clients"]
            self.federation = Federation(self.config["server_config"]["fl_method"], base_model=self.model_init.get_base_model())
            self.client_acknowledge_dict = {}
            self.client_acknowledge_flag = {}
            self.client_index_map = {}
            self.client_count = 0
            self.client_index = 0
            self.rounds_limit = self.config["server_config"]["rounds"]
            self.sleep = self.config["sleep_time"]
            self.round = 0
            self.client_round = {}
            asyncio.run(self.main())
        else:
            raise BaseException("[ERROR] Server Couldn't load the config")


    async def get_client_ip(self, request):
        return request.unresolved_remote[0]
    
    class CoAP(Resource):
        def __init__(self, server_instance):
            self.server_instance = server_instance
            super().__init__()

            
        async def render_get(self, request):
            client_ip = await self.server_instance.get_client_ip(request=request)
            # if "::1" in client_adr:
            #     client_ip = client_adr
            # else:
            #     client_ip, port_number = client_adr.split(':')
            
            # if client_ip == "127.0.0.1" or client_ip == "0.0.0.0" or client_ip == "localhost":
            #     client_ip = client_adr
            if client_ip not in self.server_instance.client_index_map:
                self.server_instance.client_index += 1
                self.server_instance.client_index_map[client_ip] = self.server_instance.client_index
                self.server_instance.client_count = len(self.server_instance.client_index_map)
                self.server_instance.client_acknowledge_flag[client_ip] = False

            if client_ip not in self.server_instance.client_round:
                self.server_instance.client_round[client_ip] = 0

            end = not (self.server_instance.client_round[client_ip] < self.server_instance.rounds_limit)
            if not self.server_instance.client_acknowledge_flag[client_ip]:
                self.server_instance.client_round[client_ip] += 1

                response_obj = {"data": self.server_instance.model_init.get_base_model(),
                                "end": end,
                                "wait": False
                                }
                if end:
                    await self.server_instance.remove_client(request=request)
                payload = pickle.dumps(response_obj)
                return Message(payload=payload)

            else:
                response_obj = {"data": {},
                                "end": end,
                                "wait": True
                                }
                payload = pickle.dumps(response_obj)
                return Message(payload=payload)

        async def render_post(self, request):
            payload = request.payload
            msg = pickle.loads(payload)
            client_ip = await self.server_instance.get_client_ip(request=request)
           
            # if "::1" in client_adr:
            #     client_ip = client_adr
            # else:
            #     client_ip, port_number = client_adr.split(':')
            
            # if client_ip == "127.0.0.1" or client_ip == "0.0.0.0" or client_ip == "localhost":
            #     client_ip = client_adr



            self.server_instance.client_acknowledge_dict[client_ip] = (msg["data"])
            self.server_instance.client_acknowledge_flag[client_ip] = True
            end = not (self.server_instance.client_round[client_ip] < self.server_instance.rounds_limit)
            if not msg["end"]:
                res = self.server_instance.federate()
                if end:
                    await self.server_instance.remove_client(request=request)
                if res:    
                    response_obj = {"data": self.server_instance.model_init.get_base_model(),
                                    "end": end,
                                    "wait": False
                                    }
                    payload = pickle.dumps(response_obj)
                    return Message(payload=payload)
                else:
                    response_obj = {"data": {},
                                    "end": end,
                                    "wait": True
                                    }
                    payload = pickle.dumps(response_obj)
                    return Message(payload=payload)
            else:
                response_obj = {"data": {},
                                "end": end,
                                "wait": True
                                }
                payload = pickle.dumps(response_obj)
                return Message(payload=payload)
            

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

    async def main(self):
        try:
            server_address = (self.ip, self.port)
            site = Site()
            site.add_resource(('index',), self.CoAP(server_instance=self))
            site.add_resource(('index', 'update'), self.CoAP(server_instance=self))
            context = await Context.create_server_context(site=site, bind=server_address)

            # asyncio.Task(aiocoap.Context.create_server_context(context))

            print(f"Server listening on {self.ip}:{self.port}")
            # asyncio.get_event_loop().run_forever()
            await asyncio.Future()  #

        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down server")
        except asyncio.CancelledError:
            print("Call for Cancellation")
