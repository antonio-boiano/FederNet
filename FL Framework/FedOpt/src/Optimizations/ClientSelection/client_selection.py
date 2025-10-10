import math
import logging

logger = logging.getLogger("FedOpt")


class ClientData:
    def __init__(self, rtt_time, training_time, dataset_size, data_entropy,
                 model_improvement):
        self.rtt_time = rtt_time
        self.training_time = training_time
        self.dataset_size = dataset_size
        self.data_entropy = data_entropy  # entropy calculated on data distribution
        self.model_improvement = model_improvement * 100 # accuracy improvement


    def __repr__(self):
        return f"({self.rtt_time}, {self.training_time}, {self.dataset_size}, {self.data_entropy}, {self.model_improvement})"


def calculate_scores(client_dict):
    for address,client in client_dict.items():
        client.data_score = round(client.data_entropy * client.dataset_size,5)
        client.comm_score = round(client.rtt_time,5)
        client.train_score = round(client.model_improvement / client.training_time,5)
        logger.debug(f"Scores for client {address}: [{client.data_score},{client.comm_score},{client.train_score}]")


def get_score_sorted_clients(clients):
    if len(clients) != 0:
        calculate_scores(clients)
        # sorted arrays for each metric
        sorted_arrays = {"data_score": sorted((v.data_score for v in clients.values()), reverse=True),
                         "comm_score": sorted(v.comm_score for v in clients.values()),
                         "train_score": sorted((v.train_score for v in clients.values()), reverse=True)}

        for key, array in sorted_arrays.items():
            sorted_arrays[key] = list(dict.fromkeys(array))

        result = {}
        for key, values in clients.items():
            indices_sum = (
                    find_index(values.data_score, sorted_arrays['data_score']) +
                    find_index(values.comm_score, sorted_arrays['comm_score']) +
                    find_index(values.train_score, sorted_arrays['train_score'])
            )
            result[key] = indices_sum

        sorted_addresses = [key for key, value in sorted(result.items(), key=lambda item: item[1])]
        return sorted_addresses
    else:
        return []


def calculate_entropy(distribution):
    total_samples = sum(distribution.values())
    entropy = 0.0
    for count in distribution.values():
        p = count / total_samples
        if p > 0:
            entropy -= p * math.log2(p)
    return float(f"{entropy:.{2}g}")


def find_index(value, array):
    try:
        return array.index(value)
    except ValueError:
        return -1
