#!/usr/bin/env python3
import asyncio
import random
import time
import paho.mqtt.client as mqtt
from FedOpt.src.Federation.manager import federation_manager
from FedOpt.src.Communication.communication import *
from FedOpt.src.Optimizations.DynamicSampling.dynamic_sampling import DynamicSampling
from FedOpt.src.Optimizations.ModelPruning.prune import ModelPruning
from FedOpt.src.Optimizations.ClientSelection.client_selection import *


class Server:
    def __init__(self, config=None):
        if config is not None:
            self.protocol_name = config['protocol']
            self.ip = config["ip"]
            self.port = config["port"]
            self.rounds_limit = config["server_config"]["rounds"]
            self.dyn_sampling = config["dyn_sampling"]
            self.performance_selection = config["client_selection"]
            self.client_round = config["server_config"]["client_round"]  # num client foreach round
            self.min_client_to_start = config["server_config"]["min_client_to_start"]
            self.sleep = config["sleep_time"]

            # federation variables
            self.client_responses = {}  # contains response from client
            self.client_selected = []  # contains client selected for the next round
            self.client_sel_data = {}  # contains data for client selection
            self.last_accuracy = 0
            self.client_index_map = {}  # <address,index> used for logging
            self.client_flag = []  # array containing address for which i have received the response
            self.round_type = ""
            self.client_count = 0
            self.current_round = 0
            self.client_index = 0
            self.send_time = {}
            self.synchronicity = config["synchronicity"]
            self.client_clustering = config["client_clustering"]
            self.max_time = config["server_config"]["max_time"]
            self.max_accuracy = config["server_config"]["max_accuracy"]

            self.federation = federation_manager(config)
            self.model_data = None
            self.message_lock = asyncio.Lock()

            self.dynamic_sampling = DynamicSampling(config["decay_rate"], self.client_round)
            self.model_pruning = ModelPruning(config["prune_ratio"], self.federation.server_model.model)
            self.pruning_flag = config["model_pruning"]
            self.prune_interval = config["prune_interval"]
            self.time_to_prune = False
            self.channel_indexes_pruned = []

            self.mqtt_client = mqtt.Client(transport="websockets",protocol=mqtt.MQTTv311)

            self.new_connection_index_map = {}
            self.used_addresses = []

            self.accuracies = []
            self.file_name = config["accuracy_file"]
        else:
            raise ValueError("[ERROR] Server Couldn't load the config")

    async def main(self):
        self.start_time = time.time()
        keepalive = 1000 if __debug__ else 60
        logger.info("keepalive: " + str(keepalive))
        logger.info("Starting the server with MQTT modifications...")
        # Assign the 'on_connect' method as the callback function to handle when the MQTT client connects to the broker
        self.mqtt_client.on_connect = self.on_connect # 
        # Assign the 'on_message' method as the callback function to handle when the MQTT client receives a message
        self.mqtt_client.on_message = self.on_message # Set the callback for when a message is received
        # Connect to the MQTT broker using the specified IP, port, and a keepalive interval of 60 seconds
        self.mqtt_client.connect(self.ip, self.port, keepalive) # Connect to the remote broker

        # Subscribe to the MQTT topic "FedOpt/Model/#" to receive messages related to the model
        self.mqtt_client.subscribe("FedOpt/Model/#")
        # Start the internal MQTT network loop. This allows the client to listen for incoming messages asynchronously
        self.mqtt_client.loop_start()
        # Perform the asynchronous function .federate() while loop is running
        await self.federate()
        # After the last global epoch, disconnect and stop loop
        self.mqtt_client.disconnect()
        self.mqtt_client.loop_stop()

    def on_startup(self, app):
        logger.info("Server started, waiting for connections...")

    def on_connect(self, client, userdata, flags, rc):
        # 0: Connection successful
        logger.debug("Connected with result code " + str(rc))

    def on_message(self, client, userdata, msg):
        """Callback for message received from client"""
        receive_time = time.perf_counter()
        client_id = self.get_client_id(msg)
        message = json.loads(msg.payload)
        logger.info(f"Received a message of size {len(msg.payload)} from client {client_id} of type {message['type']}")

        # check if client is already in the list, otherwise add the new client
        if client_id not in self.client_index_map:
            self.client_index += 1
            self.client_count += 1
            self.client_index_map[client_id] = self.client_index
            self.new_connection_index_map[client_id] = self.client_index
            logger.info(f"New connection from client {client_id}")
            reply = create_typed_message("index", self.client_index)
            self.mqtt_client.publish("FedOpt/ModelUpdate/" + f"{client_id}", json.dumps(reply))

        if message["type"] in ["data", "prune"]:  
            self.client_flag.append(client_id) 
            self.client_responses[client_id] = message["payload"] 
            total_time = receive_time - self.send_time[client_id]
            if "client_time" in message["payload"]:
                communication_time = round(total_time - message["payload"]['client_time'], 6)
                logger.info(
                    f"Round {self.current_round}, time for {client_id} is {communication_time}")
            if "client_info" in message["payload"]:  
                client_data = message["payload"]["client_info"]
                self.client_sel_data[client_id] = ClientData(round(total_time - client_data['client_time'], 7),
                                                                  round(client_data['training_time'], 5),
                                                                  client_data["data_size"],
                                                                  calculate_entropy(client_data["data_dist"]),
                                                                  round(client_data['accuracy'] - self.last_accuracy,
                                                                        6))
                logger.debug(f"Added info for client {client_id}: {self.client_sel_data[client_id]}!")

    async def federate(self):
        self.current_round += 1
        # This line is valid only for the 1st round
        while self.client_count < self.min_client_to_start or self.client_count < self.client_round:  # wait until enough client
            await asyncio.sleep(self.sleep)

        # first training round
        self.round_type = "train"
        logger.info("-- START Training --")
        self.client_selection_new(first_round=True)
        # send server model data to client selected
        await self.create_model_data_message()
        self.send_message_to(self.model_data, self.client_selected)

        while True:
            if self.synchronicity == 1:
                # Synchronous case: server waits for a fixed number of clients and then aggregates the models
                min_client_per_round, num_client_processed = self.client_round, self.client_count
            elif self.client_clustering:
                # Asynchronous case with client clustering: server aggregates asynchronously all the models received
                min_client_per_round, num_client_processed = 1, len(self.client_responses)
            else:
                # Asynchronous case w/o client clustering: server aggregates one model at time
                min_client_per_round, num_client_processed = 1, 1   

            if len(self.client_responses) >= min_client_per_round:  
                if self.synchronicity == 1:
                    if self.round_type == "train":
                        logger.info(f"START synchronous federation #{self.current_round}")
                        self.federation.apply(aggregated_dict=self.client_responses,
                                            num_clients=num_client_processed) 
                        self.last_accuracy = self.federation.server_model.evaluate()
                        logger.info(f"END Federation")
                    if self.round_type == "prune":
                        logger.info("START Model pruning aggregation")
                        self.channel_indexes_pruned = self.model_pruning.server_fed_prune(self.federation, self.client_responses)
                        logger.info("END Model pruning")
                        self.federation.server_model.evaluate()
                    self.reset_data()

                else: 
                    if self.round_type == "train":
                        logger.info(f"START asynchronous federation #{self.current_round}")
                        keys_subset = list(self.client_responses.keys())[:num_client_processed]
                        aggregated_dict = {k: self.client_responses[k] for k in keys_subset}
                        self.federation.apply(aggregated_dict=aggregated_dict, num_clients=num_client_processed+1)
                        self.last_accuracy = self.federation.server_model.evaluate()
                        logger.info(f"END Federation")
                        logger.info(f"Used addresses for federation: {keys_subset}")
                    else: 
                        if self.round_type == "prune":
                            raise Exception("Pruning is not supported in asynchronous mode")
                    self.used_addresses = []
                    for address, _ in aggregated_dict.items():
                        self.used_addresses.append(address)
                    self.reset_data_asynchronous()

                self.accuracies.append(self.last_accuracy)

                if self.max_time is not None:
                    print(self.max_time) 
                    stop_condition = time.time() - self.start_time >= self.max_time
                elif self.rounds_limit is not None:
                    stop_condition = self.current_round >= self.rounds_limit
                elif self.max_accuracy is not None:
                    stop_condition = self.last_accuracy >= self.max_accuracy
                else:
                    raise ValueError("No stopping condition defined: max_time, max_rounds or max_accuracy must be set")
                
                if stop_condition is False:
                    if (((self.current_round % self.prune_interval) == 0)
                            and self.pruning_flag and self.round_type != "prune"
                            and len(self.model_pruning.prune_layers) > 0
                            and self.current_round > 20):
                        self.round_type = "prune"
                        logger.info("-- START Pruning --")
                    else:
                        self.current_round += 1
                        self.round_type = "train"
                        logger.info("-- START Training --")
                        self.client_selection_new(first_round=False)
                else:
                    accuracy_figure(self.file_name, self.accuracies)
                    self.send_message_to(create_typed_message("end"), self.client_index_map.keys())
                    self.client_flag = []
                    logger.info("END of federation rounds!")
                    break

                # Send new model data or prune commands to the selected clients
                if self.round_type == "train":
                    await self.create_model_data_message()
                    self.send_message_to(self.model_data, self.client_selected)
                elif self.round_type == "prune":
                    self.send_message_to(create_typed_message("prune"), self.client_selected)
                    logger.debug("Sending prune commands to clients... DONE")
            await asyncio.sleep(self.sleep)

    def reset_data(self):
        self.client_responses.clear()
        self.model_data = None

    # Modification
    def reset_data_asynchronous(self):
        # self.client_responses = self.client_responses[self.client_round:]
        # self.model_data = None
        self.client_responses = {k: self.client_responses[k] for k in self.client_responses.keys() if k not in self.used_addresses}
        self.model_data = None
        self.client_flag = [k for k in self.client_flag if k not in self.used_addresses]

    async def remove_client(self, client_id):
        if client_id in self.client_selected:
            self.client_selected.remove(client_id)
        self.client_index_map.pop(client_id, None)
        self.client_count -= 1

    def get_client_id(self, message):
        return message.topic.split('/')[-1]  # also known as port

    def client_selection(self, first_round):
        self.client_flag = []
        if self.dyn_sampling and not first_round:  # avoid first sampling (start from default value)
            self.client_round = self.dynamic_sampling.number_of_clients(self.current_round)
        if self.performance_selection:
            logger.info("Client selection...")
            # Retrieve ordered address based on performance
            sorted_id = get_score_sorted_clients(self.client_sel_data)
            if len(sorted_id) > 0:
                client_ids = set(self.client_index_map.keys())
                # Identify clients with no data
                missing_id = list(client_ids - set(sorted_id))
                logger.debug(f"Missing data for clients {missing_id}")
                # Add client with missing data and truncate array
                self.client_selected = (missing_id + sorted_id)[:self.client_round]
            else:
                # no info about the client do random
                self.client_selected = random.sample(list(self.client_index_map.keys()), self.client_round)
            logger.info("Client selection... DONE")
        else:
            self.client_selected = random.sample(list(self.client_index_map.keys()), self.client_round)

    def client_selection_new(self, first_round):
        self.client_selected = []
        if self.dyn_sampling and not first_round:  # avoid first sampling (start from default value)
            if self.synchronicity!= 1:
                raise Exception("Dynamic sampling is not supported in asynchronous mode")
            else: 
                self.client_round = self.dynamic_sampling.number_of_clients(self.current_round)
                
        elif self.performance_selection:
            if self.synchronicity!= 1:
                raise Exception("Client selection is not supported in asynchronous mode")
            else:
                logger.info("Client selection...")
                # Retrieve ordered address based on performance
                sorted_id = get_score_sorted_clients(self.client_sel_data)
                if len(sorted_id) > 0:
                    client_ids = set(self.client_index_map.keys())
                    # Identify clients with no data
                    missing_id = list(client_ids - set(sorted_id))
                    logger.debug(f"Missing data for clients {missing_id}")
                    # Add client with missing data and truncate array
                    self.client_selected = (missing_id + sorted_id)[:self.client_round]
                else:
                    # no info about the client do random
                    self.client_selected = random.sample(list(self.client_index_map.keys()), self.client_round)
                logger.info("Client selection... DONE")
        else:               
            if self.synchronicity == 1:
                self.client_flag = []
                self.client_selected = random.sample(list(self.client_index_map.keys()), self.client_round)
            else:
                new_connection_index_map = self.new_connection_index_map.copy()
                self.client_selected = list(new_connection_index_map) # Select all clients that have connected recently
                temp = {address for address, _ in self.client_index_map.items() if address in self.used_addresses}
                self.client_selected = self.client_selected+list(temp)
                self.new_connection_index_map = {k: v for k, v in self.new_connection_index_map.items() if k not in new_connection_index_map}
        logger.info(f"Selected clients for this round: {self.client_selected}")

    async def create_model_data_message(self):
        async with self.message_lock:
            if self.model_data is None:
                logger.debug("Creating message containing server model!")
                server_model = self.federation.get_server_model()
                if len(self.channel_indexes_pruned) != 0:
                    server_model["pruning_info"] = self.channel_indexes_pruned
                    logger.debug("Added info about removed filters")
                self.model_data = create_typed_message("data", server_model)
                logger.debug("Creation of message.. DONE")

    def send_message_to(self, message, targets):
        payload = json.dumps(message)
        for client_id in targets:
            logger.debug(f"Sending message to client {client_id}")
            send_time = time.perf_counter()
            self.mqtt_client.publish("FedOpt/ModelUpdate/" + f"{client_id}", payload)
            self.send_time[client_id] = send_time  # used by client selection

# Plot function
import matplotlib.pyplot as plt

def accuracy_figure(filename, accuracies):
    plt.figure(figsize=(8, 5))
    plt.plot(accuracies, linestyle='-', color='b', label="Test Accuracy", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 0.9)
    # Salva il grafico come immagine 
    plt.savefig(filename)  # Puoi modificare il nome del file a tuo piacimento
    plt.close()  # Chiude il grafico, così eviti che venga visualizzato nella sessione remota
