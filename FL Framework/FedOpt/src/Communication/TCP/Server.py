import socket
import random
import time
import threading
import struct
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
            self.allowed_num_clients = config["server_config"]["max_num_clients"]
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
            self.lock = threading.Lock()
            self.run = True  # used to sto listening after federation stop
            self.synchronicity = config["synchronicity"]
            self.client_clustering = config["client_clustering"]

            self.federation = federation_manager(config)
            self.model_data = None

            self.dynamic_sampling = DynamicSampling(config["decay_rate"], self.client_round)
            self.model_pruning = ModelPruning(config["prune_ratio"], self.federation.server_model.model)
            self.pruning_flag = config["model_pruning"]
            self.prune_interval = config["prune_interval"]
            self.time_to_prune = False
            self.channel_indexes_pruned = []
            self.buffer_accuracy = [1, 2, 3, 4, 5, 6, 7, 8]

            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.settimeout(0.2)  # timeout for listening
            self.server_socket.bind((self.ip, self.port))  # bind the socket to the address (meaning that the server is listening on that port)
            self.server_socket.listen(self.allowed_num_clients)

            # Modification
            self.new_connection_index_map = {} # Map containing the index of the clients that have connected recently
            self.used_addresses = []
        else:
            raise ValueError("[ERROR] Server couldn't load the config")

    def main(self):
        try:
            logger.info("Starting server")
            logger.info("Waiting for new connections...")
            client_threads = []
            # Start a thread for federate process
            federate_thread = threading.Thread(target=self.federate)
            federate_thread.start()
    
            while self.client_count < self.allowed_num_clients and self.run:
                    # Accept new connections
                    try:
                        client_socket, client_address = self.server_socket.accept()
                        self.client_count += 1
                        self.client_index += 1
                        client_thread = threading.Thread(target=self.handle_client,
                                                        args=(client_socket, client_address))
                        client_thread.start()
                        client_threads.append(client_thread)
                    except socket.timeout:
                        pass

            # Wait for all client threads to finish
            for thread in client_threads:
                thread.join()

            # Wait for federation to finish
            federate_thread.join()
            self.server_socket.close()
        except Exception:
            logger.error("Server interrupted")

    def handle_client(self, client_socket, client_address):
        try:
            client_address = self.join_address(client_address)
            self.client_index_map[client_address] = self.client_index
            self.new_connection_index_map[client_address] = self.client_index
            logger.info(f"New connection from client {client_address}")
            self.send(client_socket, create_typed_message("index", self.client_index))

            while True:
                if client_address in self.client_selected and client_address not in self.client_flag:
                    if self.round_type == "prune":
                        message = self.transceive(client_socket, create_typed_message("prune")) # server sends a "prune" message to client
                        self.add_response(client_socket, message) # Server adds client response to the list of responses
                    
                    elif self.round_type == "train":
                        acquired = self.lock.acquire(blocking=False)
                        if acquired:
                            # Prepare model data if not already cached
                            if self.model_data is None:
                                logger.debug("Creating message containing server model!")
                                server_model = self.federation.get_server_model()
                                # Add pruning information if available
                                if len(self.channel_indexes_pruned) != 0:
                                    server_model["pruning_info"] = self.channel_indexes_pruned
                                    logger.debug("Added info about removed filters")
                                # Cache the model data
                                self.model_data = create_typed_message("data", server_model)
                            
                            # Record time and send model
                            send_time = time.perf_counter()

                            self.send(client_socket, self.model_data)

                            # Release lock and get client response
                            self.lock.release()
                            message = self.receive(client_socket)
                            self.add_response(client_socket, message, send_time)

                    elif self.round_type == "end": # end of connection with client 
                        message = self.transceive(client_socket, create_typed_message("end")) # server sends an "end" message to client
                        if message["type"] == "end": # verifies if client sends an "end" message
                            self.remove_client(client_socket) # remove client from the list of connected clients
                            break
                
                time.sleep(self.sleep)
        except Exception as e:
            logger.error(f"Exception: {e}")
            client_socket.close()

    def federate(self):
        self.current_round += 1
        while self.client_count < self.min_client_to_start or self.client_count < self.client_round:
            time.sleep(self.sleep)

        # First training round
        self.round_type = "train"
        logger.info("-- START Training --")
        self.client_selection_new(first_round=True)

        while self.run:
            if self.synchronicity == 1:
                # Synchronous case: server waits for a fixed number of clients and then aggregates the models
                min_client_per_round, num_client_processed = self.client_round, self.client_round
            elif self.client_clustering:
                # Asynchronous case with client clustering: server aggregates asynchronously all the models received
                min_client_per_round, num_client_processed = 1, len(self.client_responses)
            else:
                # Asynchronous case w/o client clustering: server aggregates one model at time
                min_client_per_round, num_client_processed = 1, 1   


            if len(self.client_responses) >= min_client_per_round:
                if self.synchronicity == 1: 
                    if self.round_type == "train":
                        logger.info(f"START federation #{self.current_round}")
                        self.federation.apply(aggregated_dict=self.client_responses,
                                            num_clients=num_client_processed)
                        self.last_accuracy = self.federation.server_model.evaluate()
                        logger.info(f"END Federation")
                    elif self.round_type == "prune":
                        logger.info("START Model pruning aggregation")
                        self.channel_indexes_pruned = self.model_pruning.server_fed_prune(self.federation, self.client_responses)
                        logger.info("END Model pruning")
                        self.federation.server_model.evaluate()

                    self.reset()
                else: 
                    if self.round_type == "train":
                        logger.info(f"START synchronous federation #{self.current_round}")
                        keys_subset = list(self.client_responses.keys())[:num_client_processed]
                        aggregated_dict = {k: self.client_responses[k] for k in keys_subset}
                        self.federation.apply(aggregated_dict=aggregated_dict, num_clients=num_client_processed) # assign to server model new data
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

                if self.current_round < self.rounds_limit:
                    if (((self.current_round % self.prune_interval) == 0)
                            and self.pruning_flag and self.round_type != "prune"
                            and len(self.model_pruning.prune_layers) > 0
                            and self.current_round > 1): #20
                        self.round_type = "prune"
                        logger.info("-- START Pruning --")
                        # pruning done on previous trained client
                    else:
                        if self.check_dead(): # avoid useless calculus
                            self.current_round = self.rounds_limit
                        else:
                            self.current_round += 1
                        self.round_type = "train"
                        logger.info("-- START Training --")
                        self.client_selection_new(first_round=False)
                else:
                    self.round_type = "end"
                    self.client_selected = list(self.client_index_map.keys())
                    self.client_flag = []
                    self.run = False
            time.sleep(self.sleep)

    def send(self, client_socket, message):
        try:
            logger.debug(f"Sending data to client...")
            data = json.dumps(message).encode('utf-8')
            length = len(data)
            # Send the length of the message first
            client_socket.sendall(struct.pack('!I', length))
            # Send the actual message
            client_socket.sendall(data)
            logger.debug(f"Send data to client... DONE!")
        except socket.error as e:
            logger.error(f"Socket error while sending: {e}")
        except Exception as e:
            logger.error(f"Exception while sending data: {e}")

    def receive(self, client_socket):
        try:
            logger.debug(f"Receiving data from client...")
            # Receive the length of the message
            length_bytes = recv_all(client_socket, 4)
            length = struct.unpack('!I', length_bytes)[0]
            # Receive the actual message
            data = recv_all(client_socket, length)
            message = json.loads(data.decode('utf-8'))
            logger.debug(f"Received data from client of type {message['type']}!")
            return message
        except socket.error as e:
            logger.error(f"Socket error while receiving: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON decode error: {e}")
            raise

    def transceive(self, client_socket, message):
        self.send(client_socket, message)
        return self.receive(client_socket)

    def add_response(self, client_socket, message, send_time=0.0):
        receive_time = time.perf_counter()
        total_time = receive_time - send_time
        client_address = self.join_address(client_socket.getpeername())
        if self.run==True:
            self.client_flag.append(client_address)
            self.client_responses[client_address] = message["payload"]
            if "client_info" in message["payload"]:  # message contains info about the client
                client_data = message["payload"]["client_info"]
                self.client_sel_data[client_address] = ClientData(round(total_time - client_data['client_time'],7),
                                                                round(client_data['training_time'],5),
                                                                client_data["data_size"],
                                                                calculate_entropy(client_data["data_dist"]),
                                                                round(client_data['accuracy'] - self.last_accuracy,6))
                logger.debug(f"Added info for client {client_address}: {self.client_sel_data[client_address]}!")

    def reset(self):
        self.client_responses.clear()
        self.client_flag = []
        self.model_data = None

    # Modification
    def reset_data_asynchronous(self):
        # keys_subset = list(self.client_responses.keys())[self.num_client_processed:]
        # self.client_responses = {k: self.client_responses[k] for k in keys_subset}
        self.client_responses = {k: self.client_responses[k] for k in self.client_responses.keys() if k not in self.used_addresses}
        self.model_data = None
        self.client_flag = [k for k in self.client_flag if k not in self.used_addresses]

    def join_address(self, address_tuple):
        return ':'.join(map(str, address_tuple))

    def remove_client(self, client_socket):
        client_key = self.join_address(client_socket.getpeername())
        if client_key in self.client_selected:
            self.client_selected.remove(client_key)
        self.client_index_map.pop(client_key, None)
        self.client_count -= 1
        client_socket.close()

    def client_selection(self, first_round):
        self.client_selected = []
        if self.dyn_sampling and not first_round:  # avoid first sampling (start from default value)
            self.client_round = self.dynamic_sampling.number_of_clients(self.current_round)
        if self.performance_selection:
            logger.info("Client selection...")
            # Retrieve ordered address based on performance
            sorted_address = get_score_sorted_clients(self.client_sel_data)
            if len(sorted_address) > 0:
                client_addresses = set(self.client_index_map.keys())
                # Identify clients with no data
                missing_address = list(client_addresses - set(sorted_address))
                logger.debug(f"Missing data for clients {missing_address}")
                # Add client with missing data and truncate array
                self.client_selected = (missing_address + sorted_address)[:self.client_round]
            else:
                # no info about the client do random
                self.client_selected = random.sample(list(self.client_index_map.keys()), self.client_round)
                logger.info("Client selection... DONE")
        else:
            self.client_selected = random.sample(list(self.client_index_map.keys()), self.client_round)

    ### Modification
    def client_selection_new(self, first_round):
        self.client_selected = []
        if self.synchronicity == 1: # If the server is synchronous
            self.client_flag = []
            self.client_selected = random.sample(list(self.client_index_map.keys()), self.client_round)
        else:
            # Get the new connection map
            new_connection_index_map = self.new_connection_index_map.copy()
            self.client_selected = list(new_connection_index_map) # Select all clients that have connected recently
            temp = {address for address, _ in self.client_index_map.items() if address in self.used_addresses}
            self.client_selected = self.client_selected+list(temp)
            # self.client_flag = [] 
            self.new_connection_index_map = {k: v for k, v in self.new_connection_index_map.items() if k not in new_connection_index_map}
            logger.info(f"Selected clients for this round: {self.client_selected}")
    def check_dead(self):
        self.buffer_accuracy.pop(0)
        self.buffer_accuracy.append(self.last_accuracy)
        if len(set(self.buffer_accuracy)) == 1:
            return True
        else:
            return False
