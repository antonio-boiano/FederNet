from abc import ABC, abstractmethod
from typing import Optional, Dict
from FedOpt.src.Federation.nn import *
from FedOpt.src.Federation.dataset import MNIST, CIFAR10, MedMNIST
import logging

logger = logging.getLogger("FedOpt")

# supported models and datasets
dataset_classes = {
    "MNIST": MNIST,
    "CIFAR10": CIFAR10,
    "MedMNIST": MedMNIST,
}

model_classes = {
    "MNIST": {
        "AlexNet": AlexNetMNIST,
        "LeNet5": LeNet5,
        "MLP": MLP,
    },
    "CIFAR10": {
        "VGG": VGG,
        "AlexNet": AlexNetCIFAR10,
    },
    "MedMNIST": {
        "VGG": VGGPathMNIST,
        "AlexNet": AlexNetPathMNIST,
        "AlexBig": AlexNetPathMNISTBig,
        "LeNet" : LeNetPathMNIST,
        "LeNetModified" : LeNetPathMNISTModified
    }
}


class AbstractAggregation(ABC):
    """
    Abstract base class for federation algorithm needed to be implemented by each federated aggregation algorithm. Contains server model and aggregation function
    """

    @abstractmethod
    def apply(self, aggregated_dict: Optional[Dict[str, any]] = None, num_clients: Optional[int] = None):
        """
        Apply the FedAverage to aggregated_dict. (called by Server)

        Parameters:
        - aggregated_dict (dict or None): The dictionary containing aggregated data.
        """
        pass

    @abstractmethod
    def get_server_model(self):
        """
        Get the server model.
        """
        pass

    # def set_server_model not used since the server model is updated directly in the apply function


class AbstractModel(ABC):
    """
    Abstract class for client model. Contains client model and optimization functions
    """
    def __init__(self, config):
        dataset = config["dataset"]
        model = config["model"]
        device = config["device"]
        if dataset in dataset_classes:
            self.dataset = dataset_classes[dataset](config)
            if model in model_classes[dataset]:
                self.model = model_classes[dataset][model]()
            else:
                logger.error(f"Unsupported model type for dataset {dataset}")
        else:
            logger.error("Unsupported dataset")
        print_trainable_par(self.model)
        self.device = torch.device(device) # torch.device is used to specify where tensors and models should be allocated
        self.model = self.model.to(self.device) # Moves model to device only after both exist
        self.criterion = torch.nn.CrossEntropyLoss()  # loss function

    @abstractmethod
    def train(self, rounds, index):
        """
        Train the model.

        Parameters:
            rounds (int): Number of training rounds.
            index (int): Index of the client in the federated learning setup.
        """
        pass

    def evaluate(self):
        logger.info("Evaluate model...")
        if self.dataset is not None:
            self.model.eval()
            with torch.no_grad():
                correct_predictions = total_samples = 0
                for inputs, targets in self.dataset.test_loader:
                    if isinstance(self.dataset, MedMNIST):
                        # Bug on squeeze torch.Size([1, 1])
                        if targets.shape != torch.Size([1, 1]):
                            targets = targets.squeeze().long()
                        else:
                            targets = torch.tensor(targets[:, 0])
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    predictions = self.model(inputs)
                    _, predicted = torch.max(predictions, 1)
                    correct_predictions += torch.eq(predicted, targets).sum().item()
                    total_samples += targets.size(0)

                # Backward pass and optimization
                accuracy = correct_predictions / total_samples

                logger.info(f"ACCURACY: {accuracy:.4f}")
        else:
            raise Exception("[ERROR] Dataset is None")
        logger.info("Evaluation model... DONE!")
        return accuracy

    @abstractmethod
    def get_client_model(self):
        """
        Get the model parameters to send to the server.

        Returns:
            dict: Dictionary containing the model parameters.
        """
        pass

    @abstractmethod
    def set_client_model(self, msg):
        """
        Set the model parameters received from the server.

        Parameters:
            msg (dict): Dictionary containing the model parameters.
        """
        pass


def print_trainable_par(model):
    conv_params = 0
    fc_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_params += sum(p.numel() for p in module.parameters() if p.requires_grad)
        elif isinstance(module, nn.Linear):
            fc_params += sum(p.numel() for p in module.parameters() if p.requires_grad)

    total_params = conv_params + fc_params
    logger.debug(f"Convolutional layers parameters: {conv_params}")
    logger.debug(f"Fully connected layers parameters: {fc_params}")
    logger.debug(f"Total parameters: {total_params}")