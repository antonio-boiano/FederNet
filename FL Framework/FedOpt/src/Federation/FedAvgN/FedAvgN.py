import torch
import time
from typing import Optional, Dict
from FedOpt.src.Utils.Decorators.timer import normal_timer
from FedOpt.src.Federation.dataset import MedMNIST
from FedOpt.src.Federation.abstract import AbstractAggregation, AbstractModel
from FedOpt.src.Communication.communication import state_dict_to_json,json_to_state_dict,tensor_list_to_json,json_to_tensor_list
import logging

logger = logging.getLogger("FedOpt")


class FedAvgN(AbstractAggregation):

    def __init__(self,config):
        self.device = config["device"]
        self.server_model = FedAvgNModel(config) # same model as client

        # FedAvg parameters
        self.base_model = self.server_model.model # state dict server model

    def apply(self, aggregated_dict: Optional[Dict[str, any]] = None, num_clients: Optional[int] = None):
        try:
            avg_y = [torch.zeros_like(param, device=self.device) for param in self.base_model.parameters()]
            clients_round = len(aggregated_dict)

            with torch.no_grad():
                if aggregated_dict is not None:
                    for _, msg in aggregated_dict.items():
                        for a_y, y in zip(avg_y, json_to_tensor_list(msg["client_model"], self.device)):
                            a_y.data.add_(y.data / clients_round)

                    for param, a_y in zip(self.base_model.parameters(), avg_y):
                        param.data = a_y.data

                else:
                    raise ValueError("aggregated_dict cannot be None.")
        except Exception as e:
            # Handle other exceptions if necessary
            logger.error(f"Error in apply: {e}")

    def get_server_model(self):
        return {"server_model": state_dict_to_json(self.base_model.state_dict())}


class FedAvgNModel(AbstractModel):

    def __init__(self, config):
        self.name = "FedAvgN"
        logger.info(f"Creation of {self.name} model ...")
        super().__init__(config)
        self.lr = config["client_config"]["local_step_size"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    @normal_timer
    def train(self, epochs, index):
        logger.info("Training...")
        start_time = time.perf_counter()
        self.model.train()
        if self.dataset is not None:
            for epoch in range(epochs):
                ls = []
                for inputs, targets in self.dataset.train_loader[index % self.dataset.num_parts]:
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
                logger.info(f"EPOCH: {epoch + 1}/{epochs}, LOSS: {torch.tensor(ls).mean():.4f}")
        else:
            raise Exception("Dataset is None")
        end_time = time.perf_counter()
        logger.info("Training... DONE!")
        return end_time - start_time

    def get_client_model(self):  # model params to server
        return {"client_model": tensor_list_to_json(self.model.parameters())}

    def set_client_model(self, msg):  # loading params from server
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
        self.model = self.model.to(self.device)