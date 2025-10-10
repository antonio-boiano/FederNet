#!/usr/bin/env python3
import socket
import time
import struct
from collections import defaultdict
from copy import deepcopy
from FedOpt.src.Federation.manager import model_manager
from FedOpt.src.Communication.communication import *
from FedOpt.src.Optimizations.ModelPruning.prune import *



class Client:
    def __init__(self, config=None):
        if config is not None:
            ip = config["ip"]
            port = config["port"]
            self.device = config["device"]
            self.address = (ip, port)
            self.epochs = config["client_config"]["epochs"]

            self.client_selection = config["client_selection"]

            self.model_mng = model_manager(config)  # it depends on federation algorithms
            self.model_pruning = ModelPruning(config["prune_ratio"], self.model_mng.model)
            self.index = 0

            # Dataset information
            self.num_samples = 0
            self.data_dist = {}

        else:
            raise ValueError("[ERROR] Client Couldn't load the config")

    def main(self):
        try:
            self.handle_server()
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt received, shutting down client")

    def handle_server(self):
        try:
            # Create a TCP/IP socket
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
                client_socket.connect(self.address) # connect to server (address refers to the server port)
                while True:
                    logger.debug("Waiting for message...")
                    message = self.receive(client_socket)
                    receive_time = time.perf_counter()
                    logger.debug("Received a message!")
                    if message["type"] == "index":
                        self.index = message["payload"]
                        logger.info(f"Received message with my index {self.index}")
                        self.analyze_dataset()
                    elif message["type"] == "data":  # receive model data
                        while not check_model_size(self.model_mng.model,message["payload"],self.device):
                            # local model and server model have different size
                            logger.info("Reduce model to be coherent with server model")
                            array_sum_of_kernel = self.model_pruning.client_fed_prune(self.model_mng)
                            old_network = deepcopy(self.model_mng.model)
                            channel_index_pruned = self.model_pruning.new_network(self.model_mng.model,array_sum_of_kernel)
                            # local model and server model have different size
                            if "pruning_info" in message["payload"]:
                                logger.debug("Information for pruning founded!")
                                update_client_variables(self.model_mng,message["payload"]["pruning_info"],old_network)
                            else:
                                logger.warning("Info about reduced filters not founded, impossible to reduce parameters!!")
                                update_client_variables(self.model_mng, channel_index_pruned)
                        ####
                        self.model_mng.set_client_model(message["payload"])
                        training_time = self.model_mng.train(self.epochs, self.index)
                        accuracy = self.model_mng.evaluate()
                        response = self.model_mng.get_client_model()
                        # add client information for selection
                        if self.client_selection:
                            response["client_info"] = {
                                "training_time": training_time,
                                "data_size": self.num_samples,
                                "data_dist": self.data_dist,
                                "accuracy": accuracy,
                                "client_time": time.perf_counter() - receive_time # time spent on client
                            }
                        self.send(client_socket, create_typed_message("data", response))
                    elif message["type"] == "prune":  # prune the model
                        result = self.model_pruning.client_fed_prune(self.model_mng)
                        logger.debug("Sending result of pruning to server")
                        self.send(client_socket, create_typed_message("prune", tensor_list_to_json(result)))
                    elif message["type"] == "end":
                        logger.debug("Received a request for END!")
                        self.send(client_socket, create_typed_message("end"))
                        break
                    else:
                        logger.warning("Unknown message type")
                    time.sleep(1)
        except Exception as e:
            logger.error(f"Exception: {e}")

    def send(self, client_socket, message):
        try:
            logger.debug(f"Sending data to server...")
            data = json.dumps(message).encode('utf-8')
            length = len(data)
            # Send the length of the message first
            client_socket.sendall(struct.pack('!I', length))
            # Send the actual message
            client_socket.sendall(data)
            logger.debug(f"Send data to server... DONE!")
        except socket.error as e:
            logger.error(f"Socket error while sending: {e}")
        except Exception as e:
            logger.error(f"Exception while sending data: {e}")

    def receive(self, client_socket):
        try:
            logger.debug(f"Receiving data from server...")
            # Receive the length of the message
            length_bytes = recv_all(client_socket, 4)
            length = struct.unpack('!I', length_bytes)[0]
            # Receive the actual message
            data = recv_all(client_socket, length)
            message = json.loads(data.decode('utf-8'))
            logger.debug(f"Received data from server of type {message['type']}!")
            return message
        except socket.error as e:
            logger.error(f"Socket error while receiving: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise

    def analyze_dataset(self):
        self.num_samples = len(self.model_mng.dataset.train_loader[
                              self.index % self.model_mng.dataset.num_parts]) * self.model_mng.dataset.train_batch_size
        # calculate data distribution
        label_occurrences = defaultdict(int)
        for images, labels in self.model_mng.dataset.train_loader[self.index % self.model_mng.dataset.num_parts]:
            for label in labels:
                label_occurrences[label.item()] += 1
        self.data_dist = dict(label_occurrences)
        self.num_samples = sum(self.data_dist.values())
        logger.debug(f"Number of samples for training: {self.num_samples}")