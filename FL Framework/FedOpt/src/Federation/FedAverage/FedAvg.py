import torch
import time
import numpy as np
from typing import Optional, Union, Dict, OrderedDict
from FedOpt.src.Utils.Decorators.timer import normal_timer
from FedOpt.src.Federation.abstract import AbstractAggregation, AbstractModel
from FedOpt.src.Federation.dataset import MedMNIST
from FedOpt.src.Communication.communication import state_dict_to_json, json_to_state_dict
import logging

logger = logging.getLogger("FedOpt")


class FedAvg(AbstractAggregation):

    def __init__(self, config):
        self.device = config["device"]
        self.server_model = FedAvgModel(config)  # same model as client

        # FedAvg parameters
        self.base_model = self.server_model.model  # state dict server model

    def average(self, selected_array, base_model) -> OrderedDict[str, Union[torch.Tensor, Dict]]:
        """
        Calculates the mean value of the values in the selected_array and updates the base_model with the average values.

        Args:
            selected_array: A list of dictionaries containing the values to be averaged.
            base_model: A dictionary containing the base model values to be updated with the average values.

        Returns:
            Mean values of the updated base_model.
        """
        # Iterate through the base_model keys and values
        for key in base_model.keys():
            # Create a list of tensors from the values in the selected_array
            values_array = [d[key] for d in selected_array if key in d.keys()]
            # Check if the value is a dictionary or a torch.Tensor
            if isinstance(base_model[key], dict):
                # Recursively call the function for nested mappings
                values_array, base_model[key] = self.average(selected_array=values_array, base_model=base_model[key])
            elif isinstance(base_model[key], torch.Tensor):
                # Update the base_model with the average value
                if len(values_array) > 0:
                    base_model[key] = torch.mean(torch.stack(values_array, dim=0).float(), dim=0)
        return base_model

    def apply(self, aggregated_dict: Optional[Dict[str, any]] = None, num_clients: Optional[int] = None):
        try:
            if aggregated_dict is not None:
                extract_models_data = {}
                for address, msg in aggregated_dict.items():
                    extract_models_data[address] = json_to_state_dict(msg["client_model"])
                np_array = np.array(list(extract_models_data.values()))
                base_model = self.average(np_array, self.base_model.state_dict())
                self.server_model.model.load_state_dict(base_model)
                self.server_model.model.to(self.device)
            else:
                raise ValueError("aggregated_dict cannot be None.")
        except Exception as e:
            # Handle other exceptions if necessary
            logger.error(f"Error in apply: {e}")

    def get_server_model(self):
        return {"server_model": state_dict_to_json(self.base_model.state_dict())}


class FedAvgModel(AbstractModel):

    def __init__(self, config):
        self.name = "FedAvg"
        logger.info(f"Creation of {self.name} model ...")
        super().__init__(config)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=config["client_config"]["local_step_size"])

    @normal_timer
    def train(self, rounds, index):
        logger.info("Training...")
        start_time = time.perf_counter()
        if self.dataset is not None:
            # Calculate the number of digits for formatting in the logs
            digits = int(torch.log10(torch.tensor(rounds))) + 1
            for epoch in range(rounds):
                ls = [] # Losses
                # Iterate through the training data for the specific client using the index
                for i, (inputs, targets) in enumerate(self.dataset.train_loader[index % self.dataset.num_parts]):
                    if isinstance(self.dataset, MedMNIST):
                        # Handle bug related to the target shape (torch.Size([1, 1]))
                        if targets.shape != torch.Size([1, 1]):
                            targets = targets.squeeze().long() # Remove unnecessary dimensions and convert to long
                        else:
                            targets = torch.tensor(targets[:, 0]) # Extract values from the target
                    # Forward pass: get predictions from the model
                    predictions = self.model(inputs.to(self.device))
                    # Compute the loss between predictions and targets
                    loss = self.criterion(predictions, targets.to(self.device))
                    ls.append(loss)
                    # Backward pass: zero the gradients, perform backpropagation, and update the model
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                logger.info(f"EPOCH: {epoch + 1:0{digits}d}/{rounds}, LOSS: {torch.tensor(ls).mean():.4f}")
        else:
            raise Exception("Dataset is None")
        end_time = time.perf_counter()
        logger.info("Training... DONE!")
        return end_time - start_time

    def get_client_model(self):  # model params to server
        return {"client_model": state_dict_to_json(self.model.state_dict())}

    def set_client_model(self, msg):  # loading params from server
        self.model.load_state_dict(json_to_state_dict(msg["server_model"]))
        self.model = self.model.to(self.device)
