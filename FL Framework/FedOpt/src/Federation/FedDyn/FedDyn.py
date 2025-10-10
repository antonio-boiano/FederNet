from copy import deepcopy
import torch
import time
from typing import Optional, Dict
from FedOpt.src.Utils.Decorators.timer import normal_timer
from FedOpt.src.Federation.abstract import AbstractAggregation, AbstractModel
from FedOpt.src.Federation.dataset import MedMNIST
from FedOpt.src.Communication.communication import state_dict_to_json, json_to_state_dict
import logging

logger = logging.getLogger("FedOpt")


class FedDyn(AbstractAggregation):

    def __init__(self, config):
        self.device = config["device"]
        self.server_model = FedDynModel(config)  # Same model as client

        # FedDyn parameters
        self.lr = config["server_config"]["global_step_size"]
        self.alpha = config["client_config"]["alpha"]
        self.h = {
            key: torch.zeros(params.shape, device=self.device)
            for key, params in self.server_model.model.state_dict().items()
        }

    def apply(self, aggregated_dict: Optional[Dict[str, any]] = None, num_clients: Optional[int] = None):
        """Updates the global model (x)"""

        try:
            if aggregated_dict is not None:
                with torch.no_grad():
                    # Aggregate the model updates from clients
                    thetas = [json_to_state_dict(msg["client_model"],self.device) for msg in aggregated_dict.values()]
                    num_round_clients = len(thetas)
                    server_params = self.server_model.model.state_dict()

                    self.h = {
                        key: prev_h
                        - (self.alpha / num_clients) * sum(theta[key] - old_params for theta in thetas)
                        for (key, prev_h), old_params in zip(self.h.items(), server_params.values())
                    }

                    new_parameters = {
                        key: (1 / num_round_clients) * sum(theta[key] for theta in thetas)
                        for key in server_params.keys()
                    }

                    new_parameters = {
                        key: params - (1 / self.alpha) * h_params
                        for key, (params, h_params) in
                        zip(new_parameters.keys(), zip(new_parameters.values(), self.h.values()))
                    }

                    self.server_model.model.load_state_dict(new_parameters)
            else:
                raise ValueError("aggregated_dict cannot be None.")
        except Exception as e:
            # Handle exceptions
            logger.error(f"Error in apply: {e}")

    def get_server_model(self):
        return {"server_model": state_dict_to_json(self.server_model.model.state_dict())}


class FedDynModel(AbstractModel):

    def __init__(self, config):
        self.name = "FedDyn"
        logger.info(f"Creation of {self.name} model ...")
        super().__init__(config)
        self.lr = config["client_config"]["local_step_size"]
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        # FedDyn variables
        self.alpha = config["client_config"]["alpha"]
        self.server_model = deepcopy(self.model)

        size = sum(p.numel() for p in self.server_model.parameters() if p.requires_grad)
        self.prev_grads = torch.zeros((size,), device=self.device)

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
            self.server_model = deepcopy(self.model).to(self.device)  # Initialize server model variable [Algorithm line no:7]
            global_params = self.server_model.state_dict()
            model_params = dict(self.model.named_parameters())
            par_flat = torch.cat([global_params[k].reshape(-1) for k in model_params.keys()])

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

                    # Dynamic Regularisation
                    curr_flat = torch.cat([p.reshape(-1) for p in self.model.parameters()])

                    # Due to pruning self.prev_grads can have different size
                    if self.prev_grads.size() != curr_flat.size():
                        logger.error(f"Founded different size between curr_params and prev_grads.")

                    # Compute the linear penalty
                    linear_penalty = torch.sum(self.prev_grads * curr_flat)

                    # Compute the quadratic penalty: (alpha / 2) * || curr_flat - par_flat || ^ 2
                    norm_penalty = (self.alpha / 2) * torch.linalg.norm(curr_flat - par_flat, 2) ** 2

                    # Compute the total mini-batch loss
                    loss = loss - linear_penalty + norm_penalty
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                    self.optimizer.step()
                    ls.append(loss)
                avg_loss = torch.tensor(ls).mean()
                logger.info(f"EPOCH: {epoch+1}/{epochs}, LOSS: {avg_loss:.4f}")

            cur_flat = torch.cat([p.detach().reshape(-1) for p in self.model.parameters()])
            self.prev_grads -= self.alpha * (cur_flat - par_flat)
        else:
            raise Exception("Dataset is None")
        end_time = time.perf_counter()
        logger.info("Training... DONE!")
        return end_time - start_time

    def get_client_model(self):
        return {"client_model": state_dict_to_json(self.model.state_dict())}

    def set_client_model(self, msg):
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
        self.model = self.model.to(self.device)
