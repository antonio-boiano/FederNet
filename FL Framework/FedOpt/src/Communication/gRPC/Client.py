import socket
import time
import traceback
from copy import deepcopy
from collections import defaultdict
from google.protobuf import empty_pb2
from FedOpt.src.Federation.manager import model_manager
from FedOpt.src.Communication.communication import *
from FedOpt.src.Optimizations.ModelPruning.prune import *
import grpc, asyncio
from FedOpt.src.Communication.gRPC import FedOpt_pb2,FedOpt_pb2_grpc

logger = logging.getLogger("FedOpt")

TIMEOUT_GRPC   = 120           # upload medio + margine
RECONNECT_MAX  = 60            # back-off massimo
KEEPALIVE_TIME = 60*1000       # ms, ping ogni 60 s
KEEPALIVE_TO   = 20*1000       # ms, timeout del ping
MAX_RETRY = 5

async def safe_rpc(coro):
    "Esegue una RPC con retry esponenziale su UNAVAILABLE / DEADLINE."
    back = 1
    while True:
        try:
            return await coro()
        except grpc.aio.AioRpcError as e:
            if e.code() in (grpc.StatusCode.UNAVAILABLE,
                            grpc.StatusCode.DEADLINE_EXCEEDED):
                logger.warning(f"RPC persa ({e.code().name}), retry fra {back}s")
                await asyncio.sleep(back)
                back = min(back*2, RECONNECT_MAX)
            else:
                raise


class Client:
    def __init__(self, config=None):
        if config is not None:
            self.ip = config["ip"]
            self.port = config["port"]
            self.my_ip = config["my_ip"]
            self.device = config["device"]
            self.epochs = config["client_config"]["epochs"]
            self.sleep = config["sleep_time"]
            self.end_event = asyncio.Event()
            self.my_port = self.find_free_port()

            self.client_selection = config["client_selection"]

            self.model_mng = model_manager(config)
            self.model_pruning = ModelPruning(config["prune_ratio"], self.model_mng.model)
            self.index = config["client_config"]["index"]

            self.stub = None
            self.channel = None

            # Dataset information
            self.num_samples = 0
            self.data_dist = {}
            self.client = None
            self.stop_event = asyncio.Event()
            self.pending_tasks = set()
        else:
            raise ValueError("[ERROR] Server Couldn't load the config")

    def find_free_port(self):
        """
        Find a free port on the localhost.
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            free_port = s.getsockname()[1]
            s.close()
            return free_port
    
    async def main(self):
        self.client = grpc.aio.server(
            options=[
                ('grpc.max_receive_message_length', 16 * 1024 * 1024),  # 16 MB
                ('grpc.max_send_message_length', 16 * 1024 * 1024)      # 16 MB
            ]
        )
        FedOpt_pb2_grpc.add_CommunicationServicer_to_server(self.Communication(client_instance=self), self.client)
        self.client.add_insecure_port(f"{self.my_ip}:{self.my_port}")
        logger.info("Starting client")
        await self.client.start()
        # send sync message to server
        self.channel = grpc.aio.insecure_channel(
            f"{self.ip}:{self.port}",
            options=[
                ('grpc.keepalive_time_ms',          KEEPALIVE_TIME),
                ('grpc.keepalive_timeout_ms',       KEEPALIVE_TO),
                ('grpc.keepalive_permit_without_calls', 1),
                ('grpc.enable_retries',             1),
                ('grpc.initial_reconnect_backoff_ms', 1000),
                ('grpc.max_reconnect_backoff_ms',   RECONNECT_MAX*1000),
            ])
        self.channel = grpc.aio.insecure_channel(f"{self.ip}:{self.port}",)
        self.stub = FedOpt_pb2_grpc.CommunicationStub(self.channel)
        msg = create_typed_message("sync", "")
        
        # await safe_rpc(lambda: self.stub.SendToServer(
        #         FedOpt_pb2.Message(payload=json.dumps(msg),
        #                               address=self.my_ip,
        #                               port=str(self.my_port)),timeout=TIMEOUT_GRPC))
        await self.stub.SendToServer(FedOpt_pb2.Message(payload=json.dumps(msg),address=self.my_ip,port=str(self.my_port)),timeout=TIMEOUT_GRPC)
        logger.info(f"Client started on {self.my_ip}:{self.my_port}")
        
        try:
            await self.stop_event.wait()
        finally:
            # Cancel all pending tasks
            if self.pending_tasks:
                tasks = list(self.pending_tasks)
                for task in tasks:
                    if not task.done():
                        task.cancel()
                # Wait for tasks to handle cancellation
                await asyncio.gather(*tasks, return_exceptions=True)
            
            if self.channel:
                await self.channel.close()
            
            await self.client.stop(grace=0)
            await self.client.wait_for_termination()

    class Communication(FedOpt_pb2_grpc.CommunicationServicer):
        def __init__(self, client_instance):
            super().__init__()
            self.client_instance = client_instance

        async def send_to_server_task(self, message,receive_time):
            return await self.send_message(message,receive_time)
        
        async def send_message(self,message,receive_time):
            while not check_model_size(self.client_instance.model_mng.model, message["payload"], self.client_instance.device):
                if self.client_instance.stop_event.is_set():
                    logger.info("Aborting model adjustment during shutdown")
                    return
                # local model and server model have different size
                logger.info("Reduce model to be coherent with server model")
                array_sum_of_kernel = self.client_instance.model_pruning.client_fed_prune(self.client_instance.model_mng)
                old_network = deepcopy(self.client_instance.model_mng.model)
                channel_index_pruned = self.client_instance.model_pruning.new_network(self.client_instance.model_mng.model, array_sum_of_kernel)
                # local model and server model have different size
                if "pruning_info" in message["payload"]:
                    logger.debug("Information for pruning founded!")
                    update_client_variables(self.client_instance.model_mng, message["payload"]["pruning_info"], old_network)
                else:
                    logger.warning("Info about reduced filters not founded, impossible to reduce parameters!!")
                    update_client_variables(self.client_instance.model_mng, channel_index_pruned)
            self.client_instance.model_mng.set_client_model(message["payload"])
            training_time = self.client_instance.model_mng.train(self.client_instance.epochs, self.client_instance.index)
            accuracy = self.client_instance.model_mng.evaluate()
            response = self.client_instance.model_mng.get_client_model()
            response["client_time"] = time.perf_counter() - receive_time
            # add client information for selection
            if self.client_instance.client_selection:
                response["client_info"] = {
                    "training_time": training_time,
                    "data_size": self.client_instance.num_samples,
                    "data_dist": self.client_instance.data_dist,
                    "accuracy": accuracy,
                    "client_time": time.perf_counter() - receive_time  # time spent on client
                }
            data_to_send = json.dumps(create_typed_message("data", response))
            
            for attempt in range(MAX_RETRY):
                try:
                    # await safe_rpc(lambda: self.client_instance.stub.SendToServer(
                    #  FedOpt_pb2.Message(payload=data_to_send,
                    #                        address=self.client_instance.my_ip,
                    #                        port=str(self.client_instance.my_port)),
                    #  timeout=TIMEOUT_GRPC))
                    await self.client_instance.stub.SendToServer(FedOpt_pb2.Message(
                        payload=data_to_send,
                        address=self.client_instance.my_ip,
                        port=str(self.client_instance.my_port)
                    ), timeout=TIMEOUT_GRPC)
                    break                      # ok
                except grpc.aio.AioRpcError as e:
                    if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED and attempt < MAX_RETRY-1:
                        await asyncio.sleep(2 ** attempt)   # back-off 1 s, 2 s, 4 s
                    else:
                        raise

            # await self.client_instance.stub.SendToServer(FedOpt_pb2.Message(
            #     payload=data_to_send,
            #     address=self.client_instance.my_ip,
            #     port=str(self.client_instance.my_port)
            # ), timeout=TIMEOUT_GRPC)
        
        def remove_task(self,task):
            try:
                self.client_instance.pending_tasks.remove(task)
            except KeyError:
                pass
            
            try:
                task.result()
            except asyncio.CancelledError:
                logger.debug("Send task cancelled during shutdown")
            except grpc.aio.AioRpcError as e:
                if e.code() == grpc.StatusCode.UNAVAILABLE:
                    logger.warning("Server unavailable")
                    # raise TemporaryDisconnect   # vedi wrapper sotto
                    self.client_instance.stop_event.set()
                else:
                    logger.error(f"gRPC error: {e.code().name} - {e.details()}")
            except Exception as e:
                logger.error(f"Unexpected error: {str(e)}")
        
        async def SendToClient(self, request: FedOpt_pb2.Message, context: grpc.aio.ServicerContext):
            receive_time = time.perf_counter()
            logger.debug(f"Received a message from the server...")
            message = json.loads(request.payload)
            if message["type"] == "index":
                if self.client_instance.index == None:
                    self.client_instance.index = message["payload"]
                    logger.info(f"My client index is: {self.client_instance.index}, taken from server")
                else:
                    logger.info(f"My client index is: {self.client_instance.index}, server index is: {message['payload']}")
                self.client_instance.analyze_dataset()
            
            elif message["type"] == "data":  # receive model data
                send_task = asyncio.create_task(self.send_to_server_task(message, receive_time))
                send_task.add_done_callback(self.remove_task)
                self.client_instance.pending_tasks.add(send_task)
                logger.debug("Sent model to server...")
                
            elif message["type"] == "prune":  # prune the model
                # Compute data for pruning and reduce model for next training
                result = self.client_instance.model_pruning.client_fed_prune(self.client_instance.model_mng)
                data_to_send = json.dumps(create_typed_message("prune", tensor_list_to_json(result)))
                logger.debug("Sending result of pruning to server...")
                await self.client_instance.stub.SendToServer(FedOpt_pb2.Message(payload=data_to_send,address=self.client_instance.my_ip,port=str(self.client_instance.my_port)), timeout=TIMEOUT_GRPC)
            elif message["type"] == "end":
                logger.debug("Received a request for END!")

                # Cancel tasks first
                if self.client_instance.pending_tasks:
                    tasks = list(self.client_instance.pending_tasks)
                    for task in tasks:
                        if not task.done():
                            task.cancel()
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Close channel immediately
                if self.client_instance.channel:
                    await self.client_instance.channel.close()
                
                # Signal main loop to exit
                self.client_instance.stop_event.set()
                # try:
                #     self.client_instance.stop_event.set()
                #     # await self.client_instance.client.stop(grace=0)
                # except Exception as e:
                #     print("Caught exception in client stop:")
                #     # pass
                #     traceback.print_exc()  # More detailed than just `print(e)`
            else:
                logger.warning("Received corrupted message. Skipping processing.")
            return empty_pb2.Empty()

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