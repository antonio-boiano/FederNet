import numpy as np
import logging

logger = logging.getLogger("FedOpt")


class DynamicSampling:
    def __init__(self, decay_rate=0.1, initial_clients=5):
        """
        decay_rate: the rate at which the number of devices decreases, default is 0.5
        """
        self.decay_rate = decay_rate
        self.initial_clients = initial_clients

    def number_of_clients(self, num_round):
        num_devices = self.initial_clients / np.exp(self.decay_rate * num_round)
        logger.info(f"Number of clients for round {num_round} equal to {num_devices}")
        num_devices = round(num_devices)
        return num_devices if num_devices > 1 else 2  # at least 2 client
