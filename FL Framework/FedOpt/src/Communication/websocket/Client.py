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
import json
import asyncio
import threading
import queue
import base64


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

            self.mqtt_client = mqtt.Client(transport="websockets",protocol=mqtt.MQTTv311,client_id=f"client_{self.client_id}",reconnect_on_failure=True)
            ws_path = "/"
            
            try:
                self.mqtt_client.enable_logger(logger)
            except Exception:
                # Fallback: enable default logger if a custom one isn’t available
                self.mqtt_client.enable_logger()
            
        else:
            raise ValueError("[ERROR] Client Couldn't load the config")
    
    def on_publish(self, client, userdata, mid):
        logger.debug(f"Publish completed (PUBACK) mid={mid}")
        
    def on_disconnect(self, client, userdata, rc):
        if rc != 0:
            logger.warning(f"Unexpected disconnection. rc={rc}")
        else:
            logger.debug("Client disconnected successfully.")
            
    def on_log(self, client, userdata, level, buf):
        # Level is an int (see paho.mqtt.client.MQTT_LOG_*)
        logger.debug(f"MQTT log [{level}]: {buf}")
        
    async def main(self):
        keepalive = 1000 if __debug__ else 60
        logger.debug("Starting MQTT client main function...")

        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_message = self.on_message
        self.mqtt_client.on_publish = self.on_publish
        self.mqtt_client.on_disconnect = self.on_disconnect
        self.mqtt_client.on_log = self.on_log
        
        self.mqtt_client.max_inflight_messages_set(1)          
        self.mqtt_client.max_queued_messages_set(1)          
        self.mqtt_client.reconnect_delay_set(min_delay=1, max_delay=30)
        
        self._q: queue.Queue = queue.Queue()                   # ADD
        self._worker_thread = threading.Thread(
            target=self._worker, name="ClientWorker", daemon=True
        )                                                       # ADD
        self._worker_thread.start() 
        
        # Connect to the MQTT broker using the given IP address and port, with a keepalive interval of 60 seconds
        self.mqtt_client.connect(self.ip, self.port, keepalive)
        self.mqtt_client.loop_start()
        
        try:
            # Keep the client running until the end_event is set (indicating termination)
            while not self.end_event.is_set():
                time.sleep(self.sleep)
        except KeyboardInterrupt:
            logger.error("KeyboardInterrupt received, shutting down client")
        finally:
            self.mqtt_client.disconnect()
            self.mqtt_client.loop_stop()

    # Callback for when the MQTT client successfully connects to the broker
    def on_connect(self, client, userdata, flags, rc):
        logger.debug("Connected with result code " + str(rc))
        # Safe to subscribe/publish here; runs on each reconnect
        self.mqtt_client.subscribe(f"FedOpt/ModelUpdate/{self.client_id}")
        msg = create_typed_message("sync", "")
        payload = json.dumps(msg)
        self.mqtt_client.publish(f"FedOpt/Model/{self.client_id}", payload, qos=0)
        logger.info("Sent synchronization message of length: " + str(len(payload)))


    # Callback for when the MQTT client receives a new message
    def on_message(self, client, userdata, msg):
        try:
            message = json.loads(msg.payload)
        except Exception as e:
            logger.warning(f"Failed to parse message on {msg.topic}: {e}")
            return
        self._q.put((msg.topic, message))
        
    def _worker(self):
        while True:
            topic, message = self._q.get()
            try:
                self._handle_message(topic, message)
            except Exception as e:
                logger.exception(f"Error handling message on {topic}: {e}")
            finally:
                self._q.task_done()

    def _handle_message(self, topic: str, message: dict):
        receive_time = time.perf_counter()
        logger.debug("Received a new message!")

        if message.get("type") == "index":
            if self.index is None:
                self.index = message["payload"]
                logger.info(f"My client index is: {self.index}, taken from server")
            else:
                logger.info(f"My client index is: {self.index}, server index is: {message['payload']}")
            self.analyze_dataset()

        elif message.get("type") == "data":
            # === copy your existing 'data' branch here ===
            while not check_model_size(self.model_mng.model, message["payload"], self.device):
                logger.info("Reduce model to be coherent with server model")
                array_sum_of_kernel = self.model_pruning.client_fed_prune(self.model_mng)
                old_network = deepcopy(self.model_mng.model)
                channel_index_pruned = self.model_pruning.new_network(self.model_mng.model, array_sum_of_kernel)

                if "pruning_info" in message["payload"]:
                    logger.debug("Information for pruning founded!")
                    update_client_variables(self.model_mng, message["payload"]["pruning_info"], old_network)
                else:
                    logger.warning("Info about reduced filters not founded, impossible to reduce parameters!!")
                    update_client_variables(self.model_mng, channel_index_pruned)

            self.model_mng.set_client_model(message["payload"])

            if self.index is None:
                logger.warning("No client index assigned yet; defaulting to 0 for this round.")
                self.index = 0
            training_time = self.model_mng.train(self.epochs, self.index)
            accuracy = self.model_mng.evaluate()
            response = self.model_mng.get_client_model()
            logger.info(f"Client {self.client_id} finished training and evaluation with size: {len(response)} and accuracy: {accuracy:.4f}")
            response["client_time"] = time.perf_counter() - receive_time

            if self.client_selection:
                response["client_info"] = {
                    "training_time": training_time,
                    "data_size": self.num_samples,
                    "data_dist": self.data_dist,
                    "accuracy": accuracy,
                    "client_time": time.perf_counter() - receive_time
                }

            data_to_send = json.dumps(create_typed_message("data", response))
            self.publish_nonblocking(f"FedOpt/Model/{self.client_id}", data_to_send, qos=0)
            logger.info("Sending message to server... DONE of size: " + str(len(data_to_send)) + "for data_dist: " + str(self.data_dist))

        elif message.get("type") == "prune":
            result = self.model_pruning.client_fed_prune(self.model_mng)
            data_to_send = json.dumps(create_typed_message("prune", tensor_list_to_json(result)))
            self.publish_nonblocking(f"FedOpt/Model/{self.client_id}", data_to_send, qos=0)
            logger.info("Sending message to server... DONE of size: " + str(len(data_to_send)))

        elif message.get("type") == "end":
            logger.debug("Received a request for END!")
            self.mqtt_client.disconnect()
            logger.debug("End of connection... DONE!")
            self.end_event.set()

        else:
            logger.warning("Received corrupted or unknown message. Skipping processing.")

    def publish_nonblocking(self, topic: str, payload: str, qos: int = 0, retain: bool = False):
        try:
            info = self.mqtt_client.publish(topic, payload, qos=qos, retain=retain)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                rc_name = {
                    mqtt.MQTT_ERR_SUCCESS: "SUCCESS",
                    mqtt.MQTT_ERR_NOMEM: "NOMEM",
                    mqtt.MQTT_ERR_PROTOCOL: "PROTOCOL",
                    mqtt.MQTT_ERR_INVAL: "INVAL",
                    mqtt.MQTT_ERR_NO_CONN: "NO_CONN",
                    mqtt.MQTT_ERR_CONN_REFUSED: "CONN_REFUSED",
                    mqtt.MQTT_ERR_NOT_FOUND: "NOT_FOUND",
                    mqtt.MQTT_ERR_CONN_LOST: "CONN_LOST",
                    mqtt.MQTT_ERR_TLS: "TLS",
                    mqtt.MQTT_ERR_PAYLOAD_SIZE: "PAYLOAD_SIZE",
                    mqtt.MQTT_ERR_NOT_SUPPORTED: "NOT_SUPPORTED",
                    mqtt.MQTT_ERR_AUTH: "AUTH",
                    mqtt.MQTT_ERR_ACL_DENIED: "ACL_DENIED",
                    mqtt.MQTT_ERR_UNKNOWN: "UNKNOWN",
                    mqtt.MQTT_ERR_ERRNO: "ERRNO",
                    mqtt.MQTT_ERR_QUEUE_SIZE: "QUEUE_SIZE",
                }.get(info.rc, str(info.rc))
                logger.error(f"Publish immediate failure rc={rc_name} topic={topic} bytes={len(payload)}")
            else:
                logger.debug(f"Queued publish mid={info.mid} qos={qos} topic={topic} bytes={len(payload)}")
            return info
        except Exception as e:
            logger.exception(f"Exception during publish to {topic}: {e}")
            return None

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
