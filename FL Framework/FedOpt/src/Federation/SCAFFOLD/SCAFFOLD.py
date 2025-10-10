import torch
import time
from typing import Optional, Dict
from copy import deepcopy
from FedOpt.src.Utils.Decorators.timer import normal_timer
from FedOpt.src.Federation.abstract import AbstractModel, AbstractAggregation
from FedOpt.src.Federation.dataset import MedMNIST
from FedOpt.src.Communication.communication import state_dict_to_json, json_to_state_dict, tensor_list_to_json, \
    json_to_tensor_list
import logging

logger = logging.getLogger("FedOpt")


class Scaffold(AbstractAggregation):
    """
        Server-side, contains server model and aggregation function
    """

    def __init__(self, config):
        self.device = config["device"]
        self.server_model = ScaffoldModel(config)

        # SCAFFOLD parameters
        self.server_c = [torch.zeros_like(param, device=self.device) for param in self.server_model.model.parameters()]
        self.server_lr = config["server_config"]["global_step_size"]

    def apply(self, aggregated_dict: Optional[Dict[str, any]] = None, num_clients: Optional[int] = None):
        delta_y_sum = [torch.zeros_like(param, device=self.device) for param in self.server_model.model.parameters()]
        delta_c_sum = [torch.zeros_like(param, device=self.device) for param in self.server_model.model.parameters()]
        try:
            if aggregated_dict is not None:
                clients_round = len(aggregated_dict)
                for _, msg in aggregated_dict.items():
                    delta_y_sum = [dy + cdy for dy, cdy in zip(delta_y_sum, json_to_tensor_list(msg["delta_y"], self.device))]
                    delta_c_sum = [dc + cdc for dc, cdc in zip(delta_c_sum, json_to_tensor_list(msg["delta_c"], self.device))]

                with torch.no_grad():
                    # Update server_model parameters using delta_y
                    lr_scaling_factor = self.server_lr / clients_round
                    for param, delta_y in zip(self.server_model.model.parameters(), delta_y_sum):
                        param.add_(delta_y * lr_scaling_factor)
                    # Update server_c using delta_c
                    for c_g, delta_c in zip(self.server_c, delta_c_sum):
                        c_g.add_(delta_c / num_clients)
            else:
                raise ValueError("aggregated_dict cannot be None.")
        except Exception as e:
            logger.error(f"Error in apply: {e}")

    def get_server_model(self):
        """
        Communicates global model and server's control variate to the participating clients
        """
        return {"server_model": state_dict_to_json(self.server_model.model.state_dict()),
                "server_control": tensor_list_to_json(self.server_c)}


class ScaffoldModel(AbstractModel):

    def __init__(self, config):
        self.name = "SCAFFOLD"
        logger.info(f"Creation of {self.name} model ...")
        super().__init__(config)
        self.lr = config["client_config"]["local_step_size"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # SCAFFOLD variables
        self.server_model = deepcopy(self.model)
        self.server_c = [torch.zeros_like(param, device=config["device"]) for param in self.server_model.parameters()]
        # Each client has its own control variate named client_c
        self.client_c = [torch.zeros_like(param, device=config["device"]) for param in self.server_model.parameters()]
        # delta_y & delta_c of a client are communicated to the central server after client_update has completed
        self.delta_y = None
        self.delta_c = None

    @normal_timer
    def train(self, epochs, index):
        """
        Trains the model on its local data. Then calculates delta_y and delta_c which are communicated back to the server.
        At last, updates the client_c.
        """
        logger.info("Training...")
        start_time = time.perf_counter()
        self.model.train()
        if self.dataset is not None:
            self.server_model = deepcopy(self.model).to(self.device)  # Initialize server model [Algorithm line no:7]
            for epoch in range(epochs):
                ls = []
                for inputs, targets in self.dataset.train_loader[index % self.dataset.num_parts]:
                    if isinstance(self.dataset, MedMNIST):
                        # Bug on squeeze torch.Size([1, 1])
                        if targets.shape != torch.Size([1, 1]):
                            targets = targets.squeeze().long()
                        else:
                            targets = torch.tensor(targets[:, 0])
                    self.optimizer.zero_grad()
                    output = self.model(inputs.to(self.device))
                    loss = self.criterion(output, targets.to(self.device))
                    loss.backward()
                    ls.append(loss)
                    for param, c_l, c_g in zip(self.model.parameters(), self.client_c, self.server_c):
                        param.grad += (-c_l + c_g).data
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                    self.optimizer.step()

                avg_loss = torch.mean(torch.stack(ls)).item()
                logger.info(f"EPOCH: {epoch + 1}/{epochs}, LOSS: {avg_loss:.4f}")

            # After local training, update client_c and calculate deltas
            with torch.no_grad():
                # Calculate delta_y which equals to y-x [Algorithm line no:13]
                self.delta_y = [y - x for y, x in zip(self.model.parameters(), self.server_model.parameters())]
                # Calculate new_client_c using client_c, server_c and delta_y [Algorithm line no:12]
                local_steps = epochs * len(self.dataset.train_loader[index % self.dataset.num_parts])
                a = 1 / local_steps * self.lr
                # Calculate delta_c which equals to new_client_c-client_c [Algorithm line no:13]
                new_client_c = [c_l - c_g - (a * diff) for c_l, c_g, diff in zip(self.client_c, self.server_c, self.delta_y)]
                self.delta_c = [n_c_l - c_l for n_c_l, c_l in zip(new_client_c, self.client_c)]
                # Update client_c with new_client_c
                self.client_c = deepcopy(new_client_c)
        else:
            raise Exception("Dataset is None")
        end_time = time.perf_counter()
        logger.info("Training... DONE!")
        return end_time - start_time

    def get_client_model(self):
        return {"delta_y": tensor_list_to_json(self.delta_y),
                "delta_c": tensor_list_to_json(self.delta_c)}

    def set_client_model(self, msg):
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
        self.model = self.model.to(self.device)
        self.server_c = json_to_tensor_list(msg["server_control"], self.device)
