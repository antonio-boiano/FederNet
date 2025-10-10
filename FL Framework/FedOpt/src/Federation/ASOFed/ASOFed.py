import math
import random
import torch
import time
from typing import Optional, Union, Dict, OrderedDict
from FedOpt.src.Federation.dataset import MedMNIST
from FedOpt.src.Utils.Decorators.timer import normal_timer
from FedOpt.src.Federation.abstract import AbstractAggregation, AbstractModel
from FedOpt.src.Communication.communication import json_to_tensor_list, state_dict_to_json, json_to_state_dict, tensor_list_to_json
from copy import deepcopy
import logging

logger = logging.getLogger("Aso-Fed")

class ASOFed(AbstractAggregation):
    """
    Server-side aggregation for asynchronous federated learning.
    Aggregates the client models according to:
    w_global_new = (1 - s) * w_global_old + s * w_client_new
    where s = 1 / (current_global_epoch - client_model_version)
    """

    def __init__(self, config):
        self.device = config["device"]
        self.server_model = ASOFedModel(config)  # Initialize the global model
        ############################ ASOFed specific ############################
        self.rho_param = config["server_config"]["asofed_rho"]
        self.global_lr_param = config["server_config"]["asofed_global_lr"]
        self.global_epoch = 0
        # self.total_train_samples = sum(len(self.server_model.dataset.train_loader[i%self.server_model.dataset.num_parts].dataset) for i in range(self.server_model.dataset.num_parts))
        self.total_train_samples = 0
        self.client_list = []

    def apply(self, aggregated_dict: Optional[Dict[str, any]] = None, num_clients: Optional[int] = None):
        try:
            if aggregated_dict is not None:
                for _, msg in aggregated_dict.items():
                    client_id = msg["client_id"]
                    client_samples = msg["client_samples"]
                    if client_id not in self.client_list:
                        self.client_list.append(client_id)
                        self.total_train_samples += client_samples
                    delta_state_dicts = json_to_state_dict(msg["delta_model"], self.device)
                    server_state_dicts = self.server_model.model.state_dict()
                    if self.global_epoch == 0:
                        lr = self.global_lr_param
                    else: 
                        lr = client_samples/self.total_train_samples
                    print('lr: ', lr)
                    # Perform the aggregation using the formula
                    with torch.no_grad():
                        new_state_dicts = {
                            key: server_params + lr * delta_params
                            for key, (server_params, delta_params) in
                            zip(server_state_dicts.keys(), zip(server_state_dicts.values(), delta_state_dicts.values()))
                        }
                        self.server_model.model.load_state_dict(new_state_dicts)
                    
                logger.info(f"Global model updated at global epoch {self.global_epoch}.")
                self.global_epoch += 1
                
            else:
                raise ValueError("aggregated_dict cannot be None.")
        except Exception as e:
            logger.error(f"Error in ASO-Fed aggregation: {e}")

    def get_server_model(self):
        return {
                "server_model": state_dict_to_json(self.server_model.model.state_dict()),
                "global_epoch": self.global_epoch, 
                }
    
class ASOFedModel(AbstractModel):
    """
    The model for clients. Each client performs local training and calculates the model updates.
    """
    def __init__(self, config):
        self.name = "ASOFed"
        logger.info(f"Creation of {self.name} model ...")
        super().__init__(config)
        self.lr = config["client_config"]["local_step_size"]
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        # ASOFed variables
        self.rho = config["client_config"]["asofed_rho"]
        self.beta = config["client_config"]["asofed_beta"]
        self.model_version = 0
        self.r = 1
        self.ave_d = 1 
        self.pre_model_version = 0
        self.joint_count = 0
        self.client_id = random.randint(0, 10000)
        self.grad_s_pre = [torch.zeros_like(param) for param in self.model.parameters()]
        self.h_pre = [torch.zeros_like(param) for param in self.model.parameters()]


    # def train2(self, rounds, index):
    #     """
    #     Train the local model on the client's dataset, compute model updates (delta),
    #     and send them to the server.
    #     """
    #     logger.info("Training...")
    #     start_time = time.perf_counter()
    #     # self.model.train()
    #     self.joint_count += 1
    #     self.ave_d = (self.ave_d * (self.joint_count - 1) + (self.model_version - self.pre_model_version)) / self.joint_count
    #     self.r = max(1, math.log10(self.ave_d+1e-20))
    #     self.pre_model_version = self.model_version

    #     if self.dataset is not None:
    #         self.server_model = deepcopy(self.model).to(self.device)
    #         global_params = self.server_model.state_dict()
    #         model_params = dict(self.model.named_parameters())
    #         par_flat = torch.cat([global_params[k].reshape(-1) for k in model_params.keys()])
    #         # Calculate the number of digits for formatting in the logs
    #         digits = int(torch.log10(torch.tensor(rounds))) + 1
    #         for epoch in range(rounds):
    #             ls = [] # Losses
    #             client_data_loader = self.dataset.train_loader[index % self.dataset.num_parts]
    #             self.num_samples = len(client_data_loader.dataset) 
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
    #                 # Dynamic Regularisation
    #                 curr_flat = torch.cat([p.reshape(-1) for p in self.model.parameters()])
    #                 # Compute the quadratic penalty: (alpha / 2) * || curr_flat - par_flat || ^ 2
    #                 norm_penalty = self.rho * torch.linalg.norm(curr_flat - par_flat, 2) ** 2
    #                 # Compute the total mini-batch loss
    #                 s = loss + norm_penalty
    #                 # Compute the gradient of the loss
    #                 s.backward()

    #                 # TODO PROVA A VEDERE IN UNWEIGHTEDMODEL SE, CON QUESTO MODO DI AGGIORNARE I PESI, OTTENGO LA STESSA ACCURACY
    #                 # Update gradients for each parameter 
    #                 for name, param in self.model.named_parameters():
    #                     if param.grad is not None:
    #                         # Line 19: Compute modified gradient
    #                         grad_xi = param.grad - self.grad_s_pre[name] + self.h_pre[name]
                            
    #                         # Line 20: Update parameters
    #                         param.data -= self.lr*self.r * grad_xi
                            
    #                         # Line 21: Update momentum term h_i
    #                         self.h_pre[name] = self.beta * self.h_pre[name] + (1 - self.beta) * param.grad
                            
    #                         # Line 22: Store current gradient
    #                         self.grad_s_pre[name] = param.grad.clone()

    #                 ls.append(s)
    #                 # grad_s = torch.autograd.grad(s, self.model.parameters())
    #                 # # grad_s = {name: param.grad.clone() for name, param in self.model.named_parameters()}
    #                 # grad_xi = {
    #                 #     key: delta_s - delta_s_pre + h_pre for key, (delta_s, delta_s_pre, h_pre) in 
    #                 #     zip(grad_s.keys(), zip(self.grad_s.values(), self.grad_s_pre.values(), self.h_pre.values()))
    #                 # }          

    #             avg_loss = torch.tensor(ls).mean()    
    #             logger.info(f"EPOCH: {epoch + 1:0{digits}d}/{rounds}, LOSS: {avg_loss:.4f}")
    #     else:
    #         raise Exception("Dataset is None")
    #     # Compute Delta
    #     self_state_dicts = self.model.state_dict()
    #     server_state_dicts = self.server_model.state_dict()
    #     self.delta_state_dicts = {key: self_state_dicts[key] - server_state_dicts[key] for key in self_state_dicts.keys()}
    #     end_time = time.perf_counter()
    #     logger.info("Training... DONE!")
    #     return end_time - start_time
    
    @normal_timer
    def train2(self, rounds, index):
        """
        Train the local model on the client's dataset, compute model updates (delta),
        and send them to the server.
        """
        logger.info("Training...")
        start_time = time.perf_counter()
        # Prepare the model for training
        self.model.train()
        # Update joint count and dynamic parameters
        self.joint_count += 1
        self.ave_d = (self.ave_d * (self.joint_count - 1) + (self.model_version - self.pre_model_version)) / self.joint_count
        self.r = max(1, math.log10(self.ave_d+1e-20))
        self.pre_model_version = self.model_version
        
        if self.dataset is not None:
            self.server_model = deepcopy(self.model).to(self.device)
            global_params = self.server_model.state_dict()
            model_params = dict(self.model.named_parameters())
            par_flat = torch.cat([global_params[k].reshape(-1) for k in model_params.keys()])
            # Calculate the number of digits for formatting in the logs
            digits = int(torch.log10(torch.tensor(rounds))) + 1
            for epoch in range(rounds):
                ls = []  # List to store losses
                # Select the client's data loader
                client_data_loader = self.dataset.train_loader[index % self.dataset.num_parts]
                self.num_samples = len(client_data_loader.dataset)
                for inputs, targets in client_data_loader:
                    if isinstance(self.dataset, MedMNIST):
                        if targets.shape != torch.Size([1, 1]):
                            targets = targets.squeeze().long()
                        else:
                            targets = torch.tensor(targets[:, 0])
                    
                    # 1. Zero Gradients: Zero out gradients before forward pass. This ensures we don't accumulate gradients from previous iterations
                    self.optimizer.zero_grad()
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # 2. Forward pass: compute predictions
                    predictions = self.model(inputs)
                    
                    # 3. Compute loss: primary loss + dynamic regularization penalty
                    loss = self.criterion(predictions, targets)
                    curr_flat = torch.cat([p.reshape(-1) for p in self.model.parameters()])
                    norm_penalty = self.rho * torch.linalg.norm(curr_flat - par_flat, 2) ** 2
                    loss = loss + norm_penalty
                    ls.append(loss)
                    
                    # 4. Backward pass: compute gradients
                    loss.backward()
                    
                    # 8. Optimizer step: Update the model parameters
                    self.optimizer.step()
                    
                    # Store the loss for logging
                    ls.append(loss.item())
                    
                # 5. Gradient Modification
                for i, (param, grad_s_pre, h_pre) in enumerate(zip(self.model.parameters(), self.grad_s_pre, self.h_pre)):
                    if param.grad is not None:
                        grad_xi = param.grad - grad_s_pre + h_pre
                        self.h_pre[i] = self.beta * h_pre + (1 - self.beta) * param.grad
                        self.grad_s_pre[i] = param.grad
                        param.grad = grad_xi
                
                # 6. Adjust Learning Rate
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.lr * self.r
                
                # 7. Clip Gradients: Prevent the exploding gradient problem
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                    
                
                # Compute and log average loss for the epoch
                avg_loss = torch.tensor(ls).mean()    
                logger.info(f"EPOCH: {epoch + 1:0{digits}d}/{rounds}, LOSS: {avg_loss:.4f}")
        else:
            raise Exception("Dataset is None")
        
        # Compute model parameter updates (delta)
        self_state_dicts = self.model.state_dict()
        server_state_dicts = self.server_model.state_dict()
        self.delta_state_dicts = {key: self_state_dicts[key] - server_state_dicts[key] for key in self_state_dicts.keys()}
        
        end_time = time.perf_counter()
        logger.info("Training... DONE!")
        return end_time - start_time

    @normal_timer
    def train3(self, rounds, index):
        """
        Train the local model on the client's dataset, compute model updates (delta),
        and send them to the server.
        """
        logger.info("Training...")
        start_time = time.perf_counter()
        self.model.train()
        self.joint_count += 1
        self.ave_d = (self.ave_d * (self.joint_count - 1) + (self.model_version - self.pre_model_version)) / self.joint_count
        self.r = max(1, math.log10(self.ave_d+1e-20))
        self.pre_model_version = self.model_version
        if self.dataset is not None:
            self.server_model = deepcopy(self.model).to(self.device)
            global_params = self.server_model.state_dict()
            model_params = dict(self.model.named_parameters())
            par_flat = torch.cat([global_params[k].reshape(-1) for k in model_params.keys()])
            # Calculate the number of digits for formatting in the logs
            digits = int(torch.log10(torch.tensor(rounds))) + 1
            for epoch in range(rounds):
                ls = [] # Losses
                client_data_loader = self.dataset.train_loader[index % self.dataset.num_parts]
                self.num_samples = len(client_data_loader.dataset) 
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
                    # Compute the gradient of the loss
                    loss.backward()
                    self.optimizer.step()
                    ls.append(loss)
                    
                    ####################### PARTE BUONA ####################
                    for i, (param, grad_s_pre, h_pre) in enumerate(zip(self.model.parameters(), self.grad_s_pre, self.h_pre)):
                        if param.grad is not None:
                            grad_xi = param.grad - grad_s_pre + h_pre
                            self.h_pre[i] = self.beta * h_pre + (1 - self.beta) * param.grad
                            self.grad_s_pre[i] = param.grad
                            param.grad = grad_xi
                    ############### FINE PARTE BUONA ########################

                    # Update parameters using the modified gradients
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                    self.optimizer.step()
                    

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
    

    @normal_timer
    def train(self, rounds, index):
        """
        Train the local model on the client's dataset, compute model updates (delta),
        and send them to the server.
        """
        logger.info("Training...")
        start_time = time.perf_counter()
        # Prepare the model for training
        self.model.train()
        # Update joint count and dynamic parameters
        self.joint_count += 1
        self.ave_d = (self.ave_d * (self.joint_count - 1) + (self.model_version - self.pre_model_version)) / self.joint_count
        self.r = max(1, math.log10(self.ave_d+1e-20))
        self.pre_model_version = self.model_version
        
        if self.dataset is not None:
            self.server_model = deepcopy(self.model).to(self.device)
            global_params = self.server_model.state_dict()
            model_params = dict(self.model.named_parameters())
            par_flat = torch.cat([global_params[k].reshape(-1) for k in model_params.keys()])
            # Calculate the number of digits for formatting in the logs
            digits = int(torch.log10(torch.tensor(rounds))) + 1
            for epoch in range(rounds):
                ls = []  # List to store losses
                # Select the client's data loader
                client_data_loader = self.dataset.train_loader[index % self.dataset.num_parts]
                self.num_samples = len(client_data_loader.dataset)
                for inputs, targets in client_data_loader:
                    if isinstance(self.dataset, MedMNIST):
                        if targets.shape != torch.Size([1, 1]):
                            targets = targets.squeeze().long()
                        else:
                            targets = torch.tensor(targets[:, 0])
                    
                    # 1. Zero Gradients: Zero out gradients before forward pass. This ensures we don't accumulate gradients from previous iterations
                    self.optimizer.zero_grad()
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    
                    # 2. Forward pass: compute predictions
                    predictions = self.model(inputs)
                    
                    # 3. Compute loss: primary loss + dynamic regularization penalty
                    loss = self.criterion(predictions, targets)
                    curr_flat = torch.cat([p.reshape(-1) for p in self.model.parameters()])
                    norm_penalty = self.rho * torch.linalg.norm(curr_flat - par_flat, 2) ** 2
                    loss = loss + norm_penalty
                    ls.append(loss)
                    
                    # 4. Backward pass: compute gradients
                    loss.backward()
                    
                    # 5. Gradient Modification
                    for i, (param, grad_s_pre, h_pre) in enumerate(zip(self.model.parameters(), self.grad_s_pre, self.h_pre)):
                        if param.grad is not None:
                            grad_xi = param.grad - grad_s_pre + h_pre
                            self.h_pre[i] = self.beta * h_pre + (1 - self.beta) * param.grad
                            self.grad_s_pre[i] = param.grad
                            param.grad = grad_xi
                    
                    # 6. Adjust Learning Rate
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = self.lr * self.r
                    
                    # 7. Clip Gradients: Prevent the exploding gradient problem
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
                    
                    # 8. Optimizer step: Update the model parameters
                    self.optimizer.step()
                    
                    # Store the loss for logging
                    ls.append(loss.item())
                
                # Compute and log average loss for the epoch
                avg_loss = torch.tensor(ls).mean()    
                logger.info(f"EPOCH: {epoch + 1:0{digits}d}/{rounds}, LOSS: {avg_loss:.4f}")
        else:
            raise Exception("Dataset is None")
        
        # Compute model parameter updates (delta)
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
                "client_samples": self.num_samples,
                "client_id": self.client_id}


    def set_client_model(self, msg):
        """
        Set the global model received from the server.
        """
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
        self.model = self.model.to(self.device)
        self.model_version = msg["global_epoch"]
        