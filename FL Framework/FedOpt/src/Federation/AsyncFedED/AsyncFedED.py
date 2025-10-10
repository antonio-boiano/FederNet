import math
import random
import torch
import time
from typing import Optional, Dict, OrderedDict
from FedOpt.src.Federation.dataset import MedMNIST
from FedOpt.src.Utils.Decorators.timer import normal_timer
from FedOpt.src.Federation.abstract import AbstractAggregation, AbstractModel
from FedOpt.src.Communication.communication import json_to_tensor_list, state_dict_to_json, json_to_state_dict, tensor_list_to_json
from copy import deepcopy
import logging

logger = logging.getLogger("AsyncFedED")

class AsyncFedED(AbstractAggregation):
    """
    Server-side aggregation for asynchronous federated learning.
    Aggregates the client models according to:
    w_global_new = (1 - s) * w_global_old + s * w_client_new
    where s = 1 / (current_global_epoch - client_model_version)
    """

    def __init__(self, config):
        self.device = config["device"]
        self.server_model = AsyncFedEDModel(config)  # Initialize the global model
        ############################ AsyncFedED specific ############################
        self.lambda_param = config["server_config"]["asyncFedED_lambda"]
        self.epsilon_param = config["server_config"]["asyncFedED_epsilon"]
        self.gamma_bar_param = config["server_config"]["asyncFedED_gamma_bar"]
        self.global_epoch = 0
        self.k_param = config["server_config"]["asyncFedED_k"]
        self.new_rounds_dict = {}  # Effective dictionary (e.g., {"client_address": round_number})

    def apply(self, aggregated_dict: Optional[Dict[str, any]] = None, num_clients: Optional[int] = None):
        try:
            if aggregated_dict is not None:
                for address, msg in aggregated_dict.items(): 
                    client_id = msg["client_id"]  
                    rounds = msg["rounds"]                
                    delta_state_dicts = json_to_state_dict(msg["delta_model"], self.device)
                    old_state_dicts = json_to_state_dict(msg["old_model"], self.device)
                    server_state_dicts = self.server_model.model.state_dict()
                    # Calculate Gamma
                    gamma = self.compute_staleness2(delta_state_dicts, old_state_dicts)
                    # Calculate the server learning rate
                    server_lr = self.lambda_param / (gamma + self.epsilon_param)
                    # Perform the aggregation using the formula
                    with torch.no_grad():
                        new_state_dicts = {
                            key: server_params + server_lr * delta_params
                            for key, (server_params, delta_params) in
                            zip(server_state_dicts.keys(), zip(server_state_dicts.values(), delta_state_dicts.values()))
                        }
                        self.server_model.model.load_state_dict(new_state_dicts)
                        self.new_rounds_dict[client_id] = self.calculate_new_rounds(gamma, rounds)
                    
                logger.info(f"Global model updated at global epoch {self.global_epoch}.")
                self.global_epoch += 1
                
            else:
                raise ValueError("aggregated_dict cannot be None.")
        except Exception as e:
            logger.error(f"Error in AsyncFedED aggregation: {e}")
    
    def calculate_new_rounds(self, gamma, rounds):
        return rounds + math.floor((self.gamma_bar_param-gamma)*self.k_param)

    def get_server_model(self):
        return {
                "server_model": state_dict_to_json(self.server_model.model.state_dict()),
                "global_epoch": self.global_epoch, 
                "new_rounds_dict": self.new_rounds_dict
                }


    def compute_staleness2(self, delta_state_dicts, old_state_dicts):
        server_state_dicts = self.server_model.model.state_dict()
        diff_state_dicts = {
                            key: server_params - old_params
                            for key, (server_params, old_params) in 
                            zip(server_state_dicts.keys(), zip(server_state_dicts.values(), old_state_dicts.values()))
                        }      
        # Extract tensors from dict
        diff_tensors = list(diff_state_dicts.values())
        delta_tensors = list(delta_state_dicts.values())
        # Concatenate the tensors
        diff_vector = torch.cat([t.flatten() for t in diff_tensors])
        delta_vector = torch.cat([t.flatten() for t in delta_tensors])
        # Calculate the norm
        diff_norm = torch.norm(diff_vector).item()
        delta_norm = torch.norm(delta_vector).item()
        # Avoid division by zero
        gamma = diff_norm / delta_norm if delta_norm > 0 else float('inf')
        return gamma


class AsyncFedEDModel(AbstractModel):
    """
    The model for clients. Each client performs local training and calculates the model updates.
    """

    def __init__(self, config):
        self.name = "AsyncFedED"
        logger.info(f"Creation of {self.name} model ...")
        super().__init__(config)
        self.lr = config["client_config"]["local_step_size"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)


        # AsyncFedED variables
        self.rounds = 0
        self.id = random.randint(0, 10000)

    @normal_timer
    def train(self, rounds, index):
        """
        Train the local model on the client's dataset, compute model updates (delta),
        and send them to the server.
        """
        logger.info("Training...")
        start_time = time.perf_counter()
        self.model.train()
        
        if self.rounds is None:
            self.rounds = rounds

        print(f"Client {self.id} is training for {self.rounds} rounds.")
            
        if self.dataset is not None:
            self.server_model = deepcopy(self.model).to(self.device)
            # Calculate the number of digits for formatting in the logs
            digits = int(torch.log10(torch.tensor(rounds))) + 1
            for epoch in range(self.rounds):
                ls = [] # Losses
                for inputs, targets in self.dataset.train_loader[index % self.dataset.num_parts]:   # Iterate over batch
                    if isinstance(self.dataset, MedMNIST):
                        # Bug on squeeze torch.Size([1, 1])
                        if targets.shape != torch.Size([1, 1]):
                            targets = targets.squeeze().long()
                        else:
                            targets = torch.tensor(targets[:, 0])
                    self.optimizer.zero_grad()
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    predictions = self.model(inputs)
                    loss = self.criterion(predictions, targets)
                    loss.backward()
                    self.optimizer.step ()
                    ls.append(loss)

                avg_loss = torch.tensor(ls).mean()    
                logger.info(f"EPOCH: {epoch + 1:0{digits}d}/{rounds}, LOSS: {avg_loss:.4f}")
        else:
            raise Exception("Dataset is None")
        # Compute Delta
        self_state_dicts = self.model.state_dict()
        server_state_dicts = self.server_model.state_dict()
        self.delta_state_dicts = {key: self_state_dicts[key] - server_state_dicts[key] for key in self_state_dicts.keys()}
        end_time = time.perf_counter()
        logger.info("Training... DONE!")
        return end_time - start_time
    
    def get_client_model(self):
        """
        Returns the updated model's weights and version information to be sent to the server.
        """
        return {"delta_model": state_dict_to_json(self.delta_state_dicts), 
                "old_model": state_dict_to_json(self.server_model.state_dict()), 
                "model_version": self.model_version, 
                "rounds": self.rounds,
                "client_id": self.id}


    def set_client_model(self, msg):
        """
        Set the global model received from the server.
        """
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
        self.model = self.model.to(self.device)
        self.model_version = msg["global_epoch"]
        self.rounds = msg["new_rounds_dict"].get(str(self.id), None)