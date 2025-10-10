#!/usr/bin/env python3
from ..model_init import ModelManager
from FedOpt.Federation.Core import Federation
import pickle
import asyncio
import pika
import time

class Server:
    def __init__(self, config=None):
        if config is not None:
            self.config = config
            self.protocol_name = "AMQP"
            print("PROTOCOL:    AMQP")
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
            self.removal_list = []
            self.removal_list_copy = []
            self.passed_time = time.time()
            asyncio.run(self.main())
        else:
            raise BaseException("[ERROR] Server Couldn't load the config")

    def get_client_ip(self, method,message):
        topic_str = str(method.routing_key)
        subtopic = topic_str.split('.')[-1]  # also known as port
        client_ip = subtopic
        return client_ip

    def on_connect(self, connection):
        print("Connected to RabbitMQ")
   
    def handle(self,channel, method_frame, header_frame, body):
        
        client_id = self.get_client_ip(method_frame,body)
        data = pickle.loads(body)

        # if "::1" in client_adr:
        #     client_id = client_adr
        # else:
        #     client_id, port_number = client_adr.split(':')

        # if client_id == "127.0.0.1" or client_id == "0.0.0.0" or client_id == "localhost":
        #     client_id = client_adr

        if client_id not in self.client_index_map:
            self.client_index += 1
            self.client_index_map[client_id] = self.client_index
            self.client_count = len(self.client_index_map)
            self.client_acknowledge_flag[client_id] = False
            print(f"[INFO] New connection from client {client_id}")

        if client_id not in self.client_round:
            self.client_round[client_id] = 0

        if data["data"]:
            self.client_acknowledge_dict[client_id] = (data["data"])
            self.client_acknowledge_flag[client_id] = True
            self.client_round[client_id] += 1

            res = self.federate()

            response_obj = {"data": self.model_init.get_base_model(),
                            "end":self.client_round[client_id] >= self.rounds_limit,
                            }
            if self.client_round[client_id] >= self.rounds_limit:
                                   self.removal_list.append(client_id)
            self.channel.basic_publish(exchange='FedOpt',
                                        routing_key=f"FedOpt.Modelupdate.{client_id}",
                                        body=pickle.dumps(response_obj))
        else:
            response_obj = {"data": self.model_init.get_base_model(),
                            "end":self.client_round[client_id] >= self.rounds_limit,
                            }
            if self.client_round[client_id] >= self.rounds_limit:
                                   self.removal_list.append(client_id)
            self.channel.basic_publish(exchange='FedOpt',
                                       routing_key=f"FedOpt.Modelupdate.{client_id}",
                                       body=pickle.dumps(response_obj))

    def reset_acknowledge(self):
        self.client_acknowledge_dict.clear()
        self.client_acknowledge_flag = self.client_acknowledge_flag.fromkeys(self.client_acknowledge_flag, False)

    def remove_client(self, client_id):
        self.client_acknowledge_dict.pop(client_id, None)
        self.client_index_map.pop(client_id, None)
        self.client_acknowledge_flag.pop(client_id, None)
        self.client_count -= 1

    def federate(self):
            # while True:
            received = list(self.client_acknowledge_flag.values())
            current_time = time.time()
            if all([self.client_count > 0, sum(received) == self.client_count]) or (current_time - self.passed_time > 10 and self.client_count >=1):
                print("Federation in process")
                print(f"[INFO] Federation with client number of {self.client_count}")
                method = self.config["server_config"]["fl_method"]
                print(f"[INFO] {method}...")
                new_params = self.federation.method.apply(aggregated_dict = self.client_acknowledge_dict)
                self.model_init.set_base_model(new_params)
                self.model_init.evaluate()
                self.reset_acknowledge()
                print(f"[INFO] {method}... DONE!")
                self.passed_time = current_time
                if len(self.removal_list) > 0 :
                    self.removal_list_copy = list(set(self.removal_list))
                    for client_id in self.removal_list_copy:
                        self.remove_client(client_id)
                        self.removal_list.remove(client_id)
                # else:
                #     await asyncio.sleep(self.sleep)
        

    def on_startup(self, app):
        print("Server started, waiting for connections...")

    async def main(self):

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.ip))
        self.channel = self.connection.channel()
        
        self.channel.exchange_declare(exchange='FedOpt', exchange_type='topic')

        for i in range(1, 11):
            # Delete existing queues
            self.channel.queue_delete(queue=f'FedOpt_Model_{i}')
            self.channel.queue_delete(queue=f'FedOpt_Modelupdate_{i}')

            # Declare new queues
            self.channel.queue_declare(queue=f'FedOpt_Model_{i}', durable=True)
            self.channel.queue_declare(queue=f'FedOpt_Modelupdate_{i}', durable=True)

            # Bind queues to exchange
            self.channel.queue_bind(exchange='FedOpt', queue=f'FedOpt_Modelupdate_{i}', routing_key=f'FedOpt.Modelupdate.{i}')
            self.channel.queue_bind(exchange='FedOpt', queue=f'FedOpt_Model_{i}', routing_key=f'FedOpt.Model.{i}')

            # Set up callback functions for each queue
            self.channel.basic_consume(queue=f'FedOpt_Model_{i}', on_message_callback=self.handle)  #, auto_ack=True

        # print(f' [*] Waiting for messages. To exit press CTRL+C')
        self.channel.start_consuming()
