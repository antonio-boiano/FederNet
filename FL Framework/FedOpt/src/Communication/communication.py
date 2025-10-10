import os
import torch
import asyncio
import logging
import json

logger = logging.getLogger("FedOpt")

dir_path = os.path.dirname(os.path.realpath(__file__))
allowed_protocols = set([folder.name for folder in os.scandir(dir_path) if os.path.isdir(folder)])
if "__pycache__" in allowed_protocols:
    allowed_protocols.remove("__pycache__")
allowed_modes = {"Client", "Server"}


class CommunicationManager:
    """
    Communication manager class that creates and handle communication between clients and servers.
    """

    def __init__(self, config):
        self.protocol_name = config["protocol"].lower()  # tcp or mqtt
        self.mode = config['mode'].lower()  # client or server
        self.config = config

    def run_instance(self):
        if all([self.protocol_name, self.mode]):
            # __________________________TCP_____________________________ #
            if self.protocol_name == "tcp":
                if self.mode == "client":
                    from FedOpt.src.Communication.TCP.Client import Clientimport 
                    import torch
                    logger.info("Launching TCP Client ")
                    mode = Client(config=self.config)
                    mode.main()
                elif self.mode == "server":
                    from FedOpt.src.Communication.TCP.Server import Server
                    logger.info("Launching TCP Server ")
                    mode = Server(config=self.config)
                    mode.main()
                else:
                    raise ValueError(f"Error. Try --mode {allowed_modes}.")

            # __________________________REST____________________________ #
            elif self.protocol_name == "rest":
                if self.mode == "client":
                    from FedOpt.src.Communication.REST.Client import Client
                    mode = Client(config=self.config)
                elif self.mode == "server":
                    from FedOpt.src.Communication.REST.Server import Server
                    mode = Server(config=self.config)
                else:
                    raise ValueError(f"Error. Try --mode {allowed_modes}.")

            # __________________________CoAP_____________________________ #
            elif self.protocol_name == "coap":
                if self.mode == "client":
                    from FedOpt.src.Communication.CoAP.Client import Client
                    mode = Client(config=self.config)
                elif self.mode == "server":
                    from FedOpt.src.Communication.CoAP.Server import Server
                    mode = Server(config=self.config)
                else:
                    raise ValueError(f"Error. Try --mode {allowed_modes}.")

            # __________________________MQTT_____________________________ #
            elif self.protocol_name == "mqtt":
                if self.mode == "client":
                    from FedOpt.src.Communication.MQTT.Client import Client
                    logger.info("Launching MQTT Client ")
                    mode = Client(config=self.config)
                    asyncio.run(mode.main())
                elif self.mode == "server":
                    from FedOpt.src.Communication.MQTT.Server import Server
                    logger.info("Launching MQTT Server ")
                    mode = Server(config=self.config)
                    asyncio.run(mode.main())
                else:
                    raise ValueError(f"Error. Try --mode {allowed_modes}.")

            # __________________________AMQP_____________________________ #
            elif self.protocol_name == "amqp":
                if self.mode == "client":
                    from FedOpt.src.Communication.AMQP.Client import Client
                    mode = Client(config=self.config)
                elif self.mode == "server":
                    from FedOpt.src.Communication.AMQP.Server import Server
                    mode = Server(config=self.config)
                else:
                    raise ValueError(f"Error. Try --mode {allowed_modes}.")

            # __________________________GRPC_____________________________ #
            elif self.protocol_name == "grpc":
                if self.mode == "client":
                    from FedOpt.src.Communication.gRPC.Client import Client
                    mode = Client(config=self.config)
                    asyncio.run(mode.main())
                elif self.mode == "server":
                    from FedOpt.src.Communication.gRPC.Server import Server
                    logger.info("Launching MQTT over Websocket Server ")
                    mode = Server(config=self.config)
                    asyncio.run(mode.main())
                else:
                    raise ValueError(f"Error. Try --mode {allowed_modes}.")
            
            # __________________________WebSocket_____________________________ #
            elif self.protocol_name == "websocket":
                if self.mode == "client":
                    from FedOpt.src.Communication.websocket.Client import Client
                    mode = Client(config=self.config)
                    asyncio.run(mode.main())
                elif self.mode == "server":
                    from FedOpt.src.Communication.websocket.Server import Server
                    logger.info("Launching WebSocket Server ")
                    mode = Server(config=self.config)
                    asyncio.run(mode.main())
                else:
                    raise ValueError(f"Error. Try --mode {allowed_modes}.")
            else:
                raise ValueError(f"Error. Try --protocol {allowed_protocols}")
        else:
            raise ValueError(f"Error. Both --protocol and --mode should be defined!")


def create_typed_message(type, payload=""):
    # possible types are 'sync','data','ack','end'
    message = {"type": type, "payload": payload}
    return message  # convert data to json string


def recv_all(client_socket, length):
    data = bytearray()
    while len(data) < length:
        packet = client_socket.recv(length - len(data))
        if not packet:
            raise ConnectionResetError("Connection reset by peer")
        data.extend(packet)
    return data


# functions because tensor is not directly serializable
def state_dict_to_json(state_dict):
    """
    Converts the state_dict to a serializable format in JSON
    """
    return json.dumps({k: v.cpu().numpy().tolist() for k, v in state_dict.items()})


def json_to_state_dict(json_data,device="cpu"):
    """
    Converts the JSON to the original state_dict format
    """
    state_dict_serializable = json.loads(json_data)
    return {k: torch.tensor(v,device=device) for k, v in state_dict_serializable.items()}


def tensor_list_to_json(tensor_list):
    return json.dumps([tensor.detach().cpu().numpy().tolist() for tensor in tensor_list])


def json_to_tensor_list(json_data, device="cpu"):
    return [torch.tensor(tensor_data, device=device) for tensor_data in json.loads(json_data)]


# struct of message
"""
{
    "type" : "data",
    "payload" : {
                    "model_data" : tensor,
                    "delta_y" : [..] #used in scaffold for example
                    "extra": ...
                }
}
"""
