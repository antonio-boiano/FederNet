#!/usr/bin/env python3
import paho.mqtt.client as mqtt
import time
import socket
import random
import sys
from copy import deepcopy
from collections import defaultdict
from FedOpt.src.Federation.manager import model_manager
from FedOpt.src.Communication.communication import *
from FedOpt.src.Optimizations.ModelPruning.prune import *


class Client:
    def __init__(self, config=None):
        if config is not None:
            self.ip = config["ip"]
            self.port = config["port"]
            self.my_ip = socket.gethostbyname(socket.gethostname())
            self.device = config["device"]
            self.epochs = config["client_config"]["epochs"]
            self.sleep = config["sleep_time"]
            self.end_event = asyncio.Event()

            self.client_selection = config["client_selection"]

            self.model_mng = model_manager(config)
            self.model_pruning = ModelPruning(config["prune_ratio"], self.model_mng.model)
            self.index = config["client_config"]["index"]

            # Dataset information
            self.num_samples = 0
            self.data_dist = {}

            localhost_range = ['127.0.0.1', 'localhost', '127.0.1.1', '::1', '0:0:0:0:0:0:0:1', '0:0:0:0:0:0:0:0', '::']
            
            if self.my_ip in localhost_range:
                logger.debug("Client IP is localhost, assigning a random client ID")
                self.client_id = random.randint(1, 10000)
            else:
                self.client_id = self.my_ip.split('.')[-1]
            logger.debug(f"Client ID: {self.client_id}")

            self.mqtt_client = mqtt.Client()
        else:
            raise ValueError("[ERROR] Client Couldn't load the config")

    async def main(self):
        keepalive = 1000 if __debug__ else 60
        logger.debug("Starting MQTT client main function...")
        # Assign the 'on_connect' callback method that handles the MQTT connection result
        self.mqtt_client.on_connect = self.on_connect
        # Assign the 'on_message' callback method that handles incoming messages
        self.mqtt_client.on_message = self.on_message
        # Connect to the MQTT broker using the given IP address and port, with a keepalive interval of 60 seconds
        self.mqtt_client.connect(self.ip, self.port, keepalive)
        
        # Create a synchronization message of type "sync" and publish it to a topic for the server
        msg = create_typed_message("sync", "")
        self.mqtt_client.publish("FedOpt/Model/" + f"{self.client_id}", json.dumps(msg))
        logger.info("Sending synchronization message to server of type 'sync'with length: " + str(len(msg)))
        
        # Subscribe to the "ModelUpdate" topic with the client-specific ID to receive updates from the server
        self.mqtt_client.subscribe("FedOpt/ModelUpdate/" + f"{self.client_id}")

        # Start the MQTT loop to handle network communication in the background
        self.mqtt_client.loop_start()
        
        try:
            # Keep the client running until the end_event is set (indicating termination)
            while not self.end_event.is_set():
                time.sleep(self.sleep)
        except KeyboardInterrupt:
            logger.error("KeyboardInterrupt received, shutting down client")
        self.mqtt_client.disconnect()
        self.mqtt_client.loop_stop()

    # Callback for when the MQTT client successfully connects to the broker
    def on_connect(self, client, userdata, flags, rc):
        logger.debug("Connected with result code " + str(rc))

    # Callback for when the MQTT client receives a new message
    def on_message(self, client, userdata, msg):
        receive_time = time.perf_counter()
        logger.debug("Received a new message!")
        message = json.loads(msg.payload)
        if message["type"] == "index":
            if self.index == None: 
                self.index = message["payload"]
                logger.info(f"My client index is: {self.index}, taken from server")
            else: 
                logger.info(f"My client index is: {self.index}, server index is: {message['payload']}")
            self.analyze_dataset()

        elif message["type"] == "data":  # receive model data
            # Check if the client model size matches the server model
            while not check_model_size(self.model_mng.model, message["payload"], self.device):
                # If the model sizes don't match, reduce the local model to be consistent with the server model (maybe because the client has joined later)
                logger.info("Reduce model to be coherent with server model")
                 # Prune the model
                array_sum_of_kernel = self.model_pruning.client_fed_prune(self.model_mng)
                old_network = deepcopy(self.model_mng.model)
                channel_index_pruned = self.model_pruning.new_network(self.model_mng.model, array_sum_of_kernel)
                
                # local model and server model have different size -> update the client model based on the aggregation algorithm
                if "pruning_info" in message["payload"]:
                    logger.debug("Information for pruning founded!")
                    update_client_variables(self.model_mng, message["payload"]["pruning_info"], old_network)
                else:
                    logger.warning("Info about reduced filters not founded, impossible to reduce parameters!!")
                    update_client_variables(self.model_mng, channel_index_pruned)
            
            # Set the new model data received from the server
            self.model_mng.set_client_model(message["payload"])
            
            ############################
            # train the model
            training_time = self.model_mng.train(self.epochs, self.index)
            # evaluate the model
            accuracy = self.model_mng.evaluate()
            response = self.model_mng.get_client_model()
            logger.info(f"Client {self.client_id} finished training and evaluation with size: {len(response)} and accuracy: {accuracy:.4f}")
            response["client_time"] = time.perf_counter() - receive_time
            # add client information for selection
            if self.client_selection:
                response["client_info"] = {
                    "training_time": training_time,
                    "data_size": self.num_samples,
                    "data_dist": self.data_dist,
                    "accuracy": accuracy,
                    "client_time": time.perf_counter() - receive_time  # time spent on client
                }
            data_to_send = json.dumps(create_typed_message("data",response))
            self.mqtt_client.publish("FedOpt/Model/" + f"{self.client_id}", data_to_send)
            logger.info("Sending message to server... DONE of size: " + str(len(data_to_send))+ "for data_dist: " + str(self.data_dist))

        elif message["type"] == "prune":  # prune the model
            # Compute data for pruning and reduce model for next training
            result = self.model_pruning.client_fed_prune(self.model_mng)
            # Send the updated model data (response) back to the server
            data_to_send = json.dumps(create_typed_message("prune", tensor_list_to_json(result)))
            self.mqtt_client.publish("FedOpt/Model/" + f"{self.client_id}", data_to_send)
            logger.info("Sending message to server... DONE of size: " + str(len(data_to_send)))
        
        elif message["type"] == "end":
            logger.debug("Received a request for END!")
            client.disconnect()
            logger.debug("End of connection... DONE!")
            self.end_event.set()
        else:
            logger.warning("Received corrupted message. Skipping processing.")

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
