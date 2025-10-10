#!/usr/bin/env python3
from ..model_init import ModelManager
import asyncio
import pika
import pickle
import random

class Client:
    def __init__(self, config=None):
        self.config = config
        self.ip = self.config["ip"]
        self.port = self.config["port"]
        self.address = f"http://{self.ip}:{self.port}"
        self.model_init = ModelManager()
        self.index = 0
        self.end = False
        self.wait = False
        self.sleep = self.config["sleep_time"]
        self.client_id = random.randint(1, 10)
        asyncio.run(self.main())

    def on_connect(self, connection):
        print("Connected to RabbitMQ")

    def on_message(self, channel, method_frame, header_frame, body):
        response = pickle.loads(body)
        self.end = response["end"]
        
        if response["data"] and not self.end: 
                self.model_init.set_base_model(response["data"])
                self.model_init.train(self.config["client_config"]["rounds"], self.index)  
                msg = {"data": self.model_init.get_base_model(),
                       "end": self.end,
                        }
                data_to_send = pickle.dumps(msg)
                self.channel.basic_publish(exchange='FedOpt',
                                       routing_key=f"FedOpt.Model.{self.client_id}",
                                       body=data_to_send)  
        elif self.end:
                self.connection.close()
        else:
             print("Received corrupted message. Skipping processing.") 
            
            

    async def main(self):

        self.connection = pika.BlockingConnection(pika.ConnectionParameters(self.ip))
        self.channel = self.connection.channel()
        msg = {"data": {},
                "end": self.end,
                }

        self.channel.basic_consume(queue=f'FedOpt_Modelupdate_{self.client_id}', on_message_callback=self.on_message)#.callback_post_respons , auto_ack=True

        self.channel.basic_publish(exchange='FedOpt',routing_key=f'FedOpt.Model.{self.client_id}', body = pickle.dumps(msg))#

        try:
            self.channel.start_consuming()
        except KeyboardInterrupt:
            print("KeyboardInterrupt received, shutting down client")

