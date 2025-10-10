import grpc
from FedOpt.src.Communication.gRPC import FedOpt_pb2,FedOpt_pb2_grpc
import time
import random
import traceback
from google.protobuf import empty_pb2
from FedOpt.src.Federation.manager import federation_manager
from FedOpt.src.Communication.communication import *
from FedOpt.src.Optimizations.DynamicSampling.dynamic_sampling import DynamicSampling
from FedOpt.src.Optimizations.ModelPruning.prune import ModelPruning
from FedOpt.src.Optimizations.ClientSelection.client_selection import *

logger = logging.getLogger("FedOpt")

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
            self.publish_done = False
            self.send_time = {}
            self.synchronicity = config["synchronicity"]
            self.client_clustering = config["client_clustering"]
            self.max_time = config["server_config"]["max_time"]

            self.federation = federation_manager(config)
            self.model_data = None
            self.message_lock = asyncio.Lock()

            self.dynamic_sampling = DynamicSampling(config["decay_rate"], self.client_round)
            self.model_pruning = ModelPruning(config["prune_ratio"], self.federation.server_model.model)
            self.pruning_flag = config["model_pruning"]
            self.prune_interval = config["prune_interval"]
            self.time_to_prune = False
            self.channel_indexes_pruned = []

            self.server = None
            
            self.new_connection_index_map = {}
            self.used_addresses = []

            self.accuracies = []
            self.file_name = config["accuracy_file"]
            self.stop_event = asyncio.Event()
        else:
            raise ValueError("[ERROR] Server Couldn't load the config")

    async def main(self):
        self.start_time = time.time()
        try:
            # Create an asynchronous gRPC server
            # The 'options' argument specifies the maximum message size that the server can receive and send
            self.server = grpc.aio.server(
                options=[
                    ('grpc.max_receive_message_length', 16 * 1024 * 1024),  # 16 MB for incoming messages
                    ('grpc.max_send_message_length', 16 * 1024 * 1024)      # 16 MB for outgoing messages
                ]
            )

            # Add the CommunicationServicer to the server
            # This binds the Communication service to the server, so it knows how to handle requests
            # `self.Communication(self)` is an instance of the Communication service that will handle requests
            FedOpt_pb2_grpc.add_CommunicationServicer_to_server(self.Communication(self), self.server)

            # Define the server address where the server will listen for incoming connections
            # This will combine the IP and port (e.g., "127.0.0.1:50051")
            listen_addr = f"{self.ip}:{self.port}"

            # Add an insecure port to the server (no SSL/TLS encryption)
            # The server will listen for incoming requests on this address
            self.server.add_insecure_port(listen_addr)

            # Log that the server is starting, including the address it will listen on
            logger.info("Starting server on %s", listen_addr)

            # Start the server asynchronously
            # This doesn't block, so the server will start handling incoming requests in the background
            await self.server.start()

            logger.info("Server started")
            await self.federate()

            # Wait for the server to keep running and processing requests
            # This will block the main method until the server is terminated (or an error occurs)
            # await self.stop_event.wait()
            # await self.server.stop(5)
            await self.server.wait_for_termination()

        except asyncio.CancelledError:
            # If the server is cancelled (perhaps due to a shutdown or interruption), log a warning
            logger.warning("Call for Cancellation")


    class Communication(FedOpt_pb2_grpc.CommunicationServicer):
        def __init__(self, server_instance):
            super().__init__()
            self.server_instance = server_instance

        # The actual gRPC method that handles incoming messages from the client
        async def SendToServer(self,request: FedOpt_pb2.Message,context: grpc.aio.ServicerContext):
            receive_time = time.perf_counter()
            logger.debug("Received a message!")

            client_ip = request.address
            client_port = request.port
            client_address = f"{client_ip}:{client_port}"

            message = json.loads(request.payload)
            # Check if this is a new client (address not seen before)
            if client_address not in self.server_instance.client_index_map:
                self.server_instance.client_index += 1
                self.server_instance.client_count += 1
                
                # This line updates the client_index_map of the Server instance (self.server_instance).
                # The client_address is used as the key, and a ClientConnection object (containing the client's IP and port) is stored as the value.
                # 
                # Difference from `self.client_index_map`:
                # - `self.client_index_map` refers to the client_index_map *directly within the current class instance* (e.g., inside the `Server` class).
                # - `self.server_instance.client_index_map` accesses the client_index_map *of the Server instance* that is passed into the Communication class.
                # This means that while `self.client_index_map` is used within the `Server` class, `self.server_instance.client_index_map` is used to interact with the same map from within the `Communication` class, which doesn't own the map itself but has access to it through the `Server` instance.
                self.server_instance.client_index_map[client_address] = ClientConnection(client_ip, client_port)
                # check
                self.server_instance.new_connection_index_map[client_address] = self.server_instance.client_index
                logger.info(f"New connection from client {client_address}")
                # send index message
                reply = create_typed_message("index", self.server_instance.client_index)
                # Stub is a client-side object that allows the client to interact with the remote server using the service methods defined in the gRPC service definition 
                stub = self.server_instance.client_index_map[client_address].stub
                stub.SendToClient(FedOpt_pb2.Message(payload=json.dumps(reply)))            
            if message["type"] in ["data", "prune"]:  # receive client model data
                # Add client to the list of clients that responded in this round
                self.server_instance.client_flag.append(client_address)
                self.server_instance.client_responses[client_address] = message["payload"]
                total_time = receive_time - self.server_instance.send_time[client_address]
                # If the client message contains "client_time", calculate communication time
                if "client_time" in message["payload"]:
                    communication_time = round(total_time - message["payload"]['client_time'],6)
                    logger.info(f"Round {self.server_instance.current_round}, time for {client_address} is {communication_time}")
                if "client_info" in message["payload"]:  # message contains info about the client
                    client_data = message["payload"]["client_info"]
                    # Add client-specific info to the server's client_sel_data (e.g., training time, accuracy, data size)
                    self.server_instance.client_sel_data[client_address] = ClientData(round(total_time - client_data['client_time'], 7),
                                                                 round(client_data['training_time'], 5),
                                                                 client_data["data_size"],
                                                                 calculate_entropy(client_data["data_dist"]),
                                                                 round(client_data['accuracy'] - self.server_instance.last_accuracy,
                                                                       6))
                    logger.debug(f"Added info for client {client_address}: {self.server_instance.client_sel_data[client_address]}!")
            # Return an empty response as a placeholder for the server's acknowledgment
            return empty_pb2.Empty()

    async def federate(self):
        self.current_round += 1
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
                
            if len(self.client_responses) >= min_client_per_round:  # models received equal to num client selected
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
                
                if self.current_round < self.rounds_limit:
                #if time.time() - self.start_time < self.max_time:
                    if (((self.current_round % self.prune_interval) == 0)
                            and self.pruning_flag and self.round_type != "prune"
                            and len(self.model_pruning.prune_layers) > 0
                            and self.current_round > 20):
                        self.round_type = "prune"
                        logger.info("-- START Pruning --")
                        # pruning done on previous trained client
                    else:
                        self.current_round += 1
                        self.round_type = "train"
                        logger.info("-- START Training --")
                        # self.client_selection(first_round=False)
                        self.client_selection_new(first_round=False)
                else:
                    accuracy_figure(self.file_name, self.accuracies)
                    self.send_message_to(create_typed_message("end"), self.client_index_map.keys())
                    self.client_flag = []
                    logger.info("END of federation rounds! " + str(self.client_index_map.keys()))
                    break
                
                if self.round_type == "train":
                    await self.create_model_data_message()
                    self.send_message_to(self.model_data, self.client_selected)
                elif self.round_type == "prune":
                    self.send_message_to(create_typed_message("prune"), self.client_selected)
                    logger.debug("Sending prune commands to clients... DONE")
                
                await asyncio.sleep(self.sleep)
            logger.debug("Stopping server...")
            await self.server.stop(5)
        #     try:
        #         await asyncio.sleep(self.sleep)
        #     except Exception as e:
        #         print("Caught exception in server AWAIT:")
        #         traceback.print_exc()  # More detailed than just `print(e)`
        # try:
        #     time.sleep(60)
        #     self.stop_event.set()
        # except Exception as e:
        #     print("Caught exception in server STOP:")
        #     traceback.print_exc()
            
    def reset_data(self):
        self.client_responses.clear()
        self.model_data = None
        
    def reset_data_asynchronous(self): # CHECK!!!! (soprattutto la flag)
        self.client_responses = {k: self.client_responses[k] for k in self.client_responses.keys() if k not in self.used_addresses}
        self.model_data = None
        self.client_flag = [k for k in self.client_flag if k not in self.used_addresses]

    async def remove_client(self, client_id):
        if client_id in self.client_selected:
            self.client_selected.remove(client_id)
        self.client_index_map.pop(client_id, None)
        self.client_count -= 1

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
        
        # #################    
        # if self.synchronicity == 1:
        #     self.client_flag = []
        #     self.client_selected = random.sample(list(self.client_index_map.keys()), self.client_round)
        # else:
        #     new_connection_index_map = self.new_connection_index_map.copy()
        #     self.client_selected = list(new_connection_index_map) # Select all clients that have connected recently
        #     temp = {address for address, _ in self.client_index_map.items() if address in self.used_addresses}
        #     self.client_selected = self.client_selected+list(temp)
        #     self.new_connection_index_map = {k: v for k, v in self.new_connection_index_map.items() if k not in new_connection_index_map}
        # logger.info(f"Selected clients for this round: {self.client_selected}")

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
        for client_address in targets:
            logger.debug(f"Sending message to client {client_address}")
            stub = self.client_index_map[client_address].stub
            send_time = time.perf_counter()
            stub.SendToClient(FedOpt_pb2.Message(payload=payload, address=self.ip,port=str(self.port)))
            self.send_time[client_address] = send_time  # used by client selection
            
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
    plt.savefig(filename)  # Puoi modificare il nome del file a tuo piacimento
    plt.close()  # Chiude il grafico, così eviti che venga visualizzato nella sessione remota 

class ClientConnection:
    def __init__(self,client_ip, client_port):
        self.channel = grpc.aio.insecure_channel(f'{client_ip}:{client_port}')
        self.stub = FedOpt_pb2_grpc.CommunicationStub(self.channel)