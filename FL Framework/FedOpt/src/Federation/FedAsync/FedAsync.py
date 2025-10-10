import torch
import time
from typing import Optional, Dict, OrderedDict
from FedOpt.src.Federation.dataset import MedMNIST
from FedOpt.src.Utils.Decorators.timer import normal_timer
from FedOpt.src.Federation.abstract import AbstractAggregation, AbstractModel
from FedOpt.src.Communication.communication import json_to_tensor_list, state_dict_to_json, json_to_state_dict, tensor_list_to_json
from copy import deepcopy
import logging

logger = logging.getLogger("FedAsync")


class FedAsync(AbstractAggregation):
    """
    Server-side aggregation for asynchronous federated learning.
    Aggregates the client models according to:
    w_global_new = (1 - s) * w_global_old + s * w_client_new
    where s = 1 / (current_global_epoch - client_model_version)
    """

    def __init__(self, config):
        self.device = config["device"]
        self.server_model = FedAsyncModel(config)  # Initialize the global model
        ############################ FedAsync specific ############################
        self.variant = config["server_config"]["fedAsync_variant"] # Hinge or Polynomial
        self.a_hinge = config["server_config"]["fedAsync_a_hinge"] # Hinge variant parameter
        self.b_hinge = config["server_config"]["fedAsync_b_hinge"] # Hinge variant parameter
        self.a_polynomial = config["server_config"]["fedAsync_a_poly"] # Polynomial variant parameter
        self.mu0 = config["server_config"]["fedAsync_mu_0"] # Regularization parameter       
        self.global_epoch = 0

    def apply(self, aggregated_dict: Optional[Dict[str, any]] = None, num_clients: Optional[int] = None):
        try:
            if aggregated_dict is not None:
                self.global_epoch += 1
                for address, msg in aggregated_dict.items():
                    client_state_dicts = json_to_state_dict(msg["client_model"], self.device)
                    server_state_dicts = self.server_model.model.state_dict()
                    model_version = msg["model_version"]
                    # Calculate aggregation factor s
                    if self.variant == "hinge":
                        if self.global_epoch <= model_version:
                            s = 1 
                        else:
                            s = 1/(self.a_hinge*(self.global_epoch - model_version-self.b_hinge)+1)
                    elif self.variant == "polynomial":
                        s = (self.global_epoch - model_version+1)**(-self.a_polynomial)
                    else:
                        raise ValueError("Invalid FedAsync variant.")
                    mu_t = self.mu0 * s

                    with torch.no_grad():
                        new_state_dicts = {
                            key:(1-mu_t)*server_params + mu_t * client_params
                            for key, (server_params, client_params) in
                            zip(server_state_dicts.keys(), zip(server_state_dicts.values(), client_state_dicts.values()))
                        }
                        
                        self.server_model.model.load_state_dict(new_state_dicts)

                logger.info(f"Global model updated at global epoch {self.global_epoch}.")
                # self.global_epoch += 1
            else:
                raise ValueError("aggregated_dict cannot be None.")
        except Exception as e:
            logger.error(f"Error in FedAsync aggregation: {e}")

    def get_server_model(self):
        return {"server_model": state_dict_to_json(self.server_model.model.state_dict()),
                "global_epoch": self.global_epoch}

class FedAsyncModel(AbstractModel):
    """
    The model for clients. Each client performs local training and calculates the model updates.
    """

    def __init__(self, config):
        self.name = "FedAsync"
        logger.info(f"Creation of {self.name} model ...")
        super().__init__(config)
        self.lr = config["client_config"]["local_step_size"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # FedAsync variables
        self.rho = config["client_config"]["fedAsync_rho"] # Regularization parameter
        self.model_version = 0
        self.server_model = deepcopy(self.model)

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
            self.server_model = deepcopy(self.model).to(self.device)
            global_params = self.server_model.state_dict()
            model_params = dict(self.model.named_parameters())
            par_flat = torch.cat([global_params[k].reshape(-1) for k in model_params.keys()])
            # Calculate the number of digits for formatting in the logs
            digits = int(torch.log10(torch.tensor(rounds))) + 1
            for epoch in range(rounds):
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
                    # Dynamic Regularisation
                    curr_flat = torch.cat([p.reshape(-1) for p in self.model.parameters()])
                    # Compute the quadratic penalty: (alpha / 2) * || curr_flat - par_flat || ^ 2
                    norm_penalty = self.rho * torch.linalg.norm(curr_flat - par_flat, 2) ** 2
                    # Compute the total mini-batch loss
                    loss = loss + norm_penalty
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                    self.optimizer.step()
                    ls.append(loss)

                avg_loss = torch.tensor(ls).mean()    
                logger.info(f"EPOCH: {epoch + 1:0{digits}d}/{rounds}, LOSS: {avg_loss:.4f}")
        else:
            raise Exception("Dataset is None")
        end_time = time.perf_counter()
        logger.info("Training... DONE!")
        return end_time - start_time
    

    def regularized_loss(self, loss, server_model):
        """
        Computes the regularized loss for the client.
        """
        server_params = list(server_model.parameters())
        client_params = list(self.model.parameters())
        param_diff = [param.data - client_param.data for param, client_param in zip(server_params, client_params)]
        diff_flattened = list()
        for layer in param_diff:
            for tensor in layer:
                if tensor.size != 0:
                    if tensor.ndimension() > 0:  # If the tensor is not a scalar
                        diff_flattened.extend(tensor.flatten().tolist())  # Flattens the tensor and adds it to the list
                    else: 
                        diff_flattened.append(tensor.item()) # If the tensor is a scalar, add it directly
        diff_flattened_tensor = torch.tensor(diff_flattened)
        norm_sq = (torch.norm(diff_flattened_tensor)**2).item()
        reg_term = 0.5 * self.rho* norm_sq
        return loss + reg_term

    def get_client_model(self):
        """
        Returns the updated model's weights and version information to be sent to the server.
        """
        return {"client_model": state_dict_to_json(self.model.state_dict()), 
                "model_version": self.model_version}



    def set_client_model(self, msg):
        """
        Set the global model received from the server.
        """
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
        self.model = self.model.to(self.device)
        self.model_version = msg["global_epoch"]