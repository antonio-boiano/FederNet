import torch
import time
from typing import Optional, Dict, OrderedDict
from FedOpt.src.Federation.dataset import MedMNIST
from FedOpt.src.Utils.Decorators.timer import normal_timer
from FedOpt.src.Federation.abstract import AbstractAggregation, AbstractModel
from FedOpt.src.Communication.communication import json_to_tensor_list, state_dict_to_json, json_to_state_dict, tensor_list_to_json
from copy import deepcopy
import logging

logger = logging.getLogger("Unweighted")

class Unweighted(AbstractAggregation):
    """
    Server-side aggregation for asynchronous federated learning.
    Aggregates the client models according to:
    w_global_new = (1 - s) * w_global_old + s * w_client_new
    where s = 1 / (current_global_epoch - client_model_version)
    """

    def __init__(self, config):
        self.device = config["device"]
        self.server_model = UnweightedModel(config)  # Initialize the global model
        ############################ Unweighted specific ############################
        self.mu0 = config["server_config"]["unweighted_mu_0"] # Regularization parameter       
        self.global_epoch = 0

    def apply(self, aggregated_dict: Optional[Dict[str, any]] = None, num_clients: Optional[int] = None):
        try:
            if aggregated_dict is not None:
                # extract_models_data = {}
                for address, msg in aggregated_dict.items():
                    client_state_dicts = json_to_state_dict(msg["client_model"], self.device)
                    server_state_dicts = self.server_model.model.state_dict()
                
                    with torch.no_grad():
                        new_state_dicts = {
                            key:(1-self.mu0)*server_params + self.mu0 * client_params
                            for key, (server_params, client_params) in
                            zip(server_state_dicts.keys(), zip(server_state_dicts.values(), client_state_dicts.values()))
                        }
                        
                        self.server_model.model.load_state_dict(new_state_dicts)

                logger.info(f"Global model updated at global epoch {self.global_epoch}.")
                self.global_epoch += 1
            else:
                raise ValueError("aggregated_dict cannot be None.")
        except Exception as e:
            logger.error(f"Error in Unweighted aggregation: {e}")

    def get_server_model(self):
        return {"server_model": state_dict_to_json(self.server_model.model.state_dict()),
                "global_epoch": self.global_epoch}


class UnweightedModel(AbstractModel):
    """
    The model for clients. Each client performs local training and calculates the model updates.
    """

    def __init__(self, config):
        self.name = "Unweighted"
        logger.info(f"Creation of {self.name} model ...")
        super().__init__(config)
        self.lr = config["client_config"]["local_step_size"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    # def train2(self, rounds, index):
    #     """
    #     Train the local model on the client's dataset, compute model updates (delta),
    #     and send them to the server.
    #     """
    #     logger.info("Training...")
    #     start_time = time.perf_counter()
    #     # TODO: Check if the model should be set to train mode
    #     self.model.train2()
    #     if self.dataset is not None:
    #         # Calculate the number of digits for formatting in the logs
    #         digits = int(torch.log10(torch.tensor(rounds))) + 1
    #         for epoch in range(rounds):
    #             ls = [] # Losses
    #             # Iterate through the training data for the specific client using the index
    #             for inputs, targets in self.dataset.train_loader[index % self.dataset.num_parts]:   # Iterate over batch
    #                 if isinstance(self.dataset, MedMNIST):
    #                     # Bug on squeeze torch.Size([1, 1])
    #                     if targets.shape != torch.Size([1, 1]):
    #                         targets = targets.squeeze().long()
    #                     else:
    #                         targets = torch.tensor(targets[:, 0])
    #                 self.optimizer.zero_grad()
    #                 inputs, targets = inputs.to(self.device), targets.to(self.device)
    #                 predictions = self.model(inputs)
    #                 loss = self.criterion(predictions, targets)
    #                 loss.backward()
    #                 for name, param in list(self.model.named_parameters()):
    #                     if param.grad is not None:
    #                         param.data -= self.lr*param.grad

    #                 # Update parameters using the modified gradients
    #                 torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
    #                 self.optimizer.step()

    #                 ls.append(loss)

    #             avg_loss = torch.tensor(ls).mean()    
    #             logger.info(f"EPOCH: {epoch + 1:0{digits}d}/{rounds}, LOSS: {avg_loss:.4f}")
    #     else:
    #         raise Exception("Dataset is None")
    #     end_time = time.perf_counter()
    #     logger.info("Training... DONE!")
    #     return end_time - start_time


    @normal_timer
    def train(self, rounds, index):
        """
        Train the local model on the client's dataset, compute model updates (delta),
        and send them to the server.
        """
        logger.info("Training...")
        start_time = time.perf_counter()
        self.model.train()
        if self.dataset is not None:
            for epoch in range(rounds):
                ls = [] # Losses
                # Iterate through the training data for the specific client using the index
                for inputs, targets in self.dataset.train_loader[index % self.dataset.num_parts]:   # Iterate over batch
                    if isinstance(self.dataset, MedMNIST):
                        # Bug on squeeze torch.Size([1, 1])
                        if targets.shape != torch.Size([1, 1]):
                            targets = targets.squeeze().long()
                        else:
                            targets = torch.tensor(targets[:, 0])
                    output = self.model(inputs.to(self.device))
                    loss = self.criterion(output, targets.to(self.device))
                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                    self.optimizer.step()
                    ls.append(loss)
                logger.info(f"EPOCH: {epoch + 1}/{rounds}, LOSS: {torch.tensor(ls).mean():.4f}")
                # avg_loss = torch.tensor(ls).mean()    
                # logger.info(f"EPOCH: {epoch + 1:0{digits}d}/{rounds}, LOSS: {avg_loss:.4f}")
        else:
            raise Exception("Dataset is None")
        end_time = time.perf_counter()
        logger.info("Training... DONE!")
        return end_time - start_time
    
    def get_client_model(self):  # model params to server
        return {"client_model": state_dict_to_json(self.model.state_dict())}
    
    def set_client_model(self, msg):
        """
        Set the global model received from the server.
        """
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
        self.model = self.model.to(self.device)