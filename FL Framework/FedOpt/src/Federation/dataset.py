import torch
from medmnist import PathMNIST
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from fedlab.utils.functional import partition_report
from fedlab.utils.dataset.partition import CIFAR10Partitioner
from fedlab.utils.dataset import MNISTPartitioner
from matplotlib import pyplot as plt
import pandas as pd
import logging
import os

logger = logging.getLogger("FedOpt")
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../Data'))
if not os.path.exists(root_path):
    os.makedirs(root_path)

class CIFAR10:

    def __init__(self, config):
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.num_parts = config["num_parts_dataset"]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        self.device = torch.device(config["device"])

        seed = config["seed"]
        alpha = config["alpha"]
        iid_ratio = config["iid_ratio"]
        test_size = config["test_size"]
        logger.debug(f"Selected alpha for dirichlet distribution: {alpha}, selected iid ratio: {iid_ratio}")

        # Load CIFAR-10 dataset
        self.training_data = datasets.CIFAR10(
            root=root_path,
            train=True,
            download=True,
            transform=self.transform
        )

        if alpha != 1:
            data_subsets = CIFAR10Partitioner(self.training_data.targets,
                                              self.num_parts,
                                              balance=None,
                                              partition="dirichlet",
                                              dir_alpha=alpha,
                                              seed=seed)
        elif iid_ratio != 1:
            data_subsets = CIFAR10Partitioner(self.training_data.targets,
                                              self.num_parts,
                                              balance=False,
                                              partition="iid",
                                              unbalance_sgm=iid_ratio,
                                              seed=seed)
        else:
            data_subsets = CIFAR10Partitioner(self.training_data.targets,
                                              self.num_parts,
                                              balance=True,
                                              partition="iid",
                                              seed=seed)

        # plot graphs about distribution of the data
        # plot_graph(self.training_data.targets, data_subsets.client_dict, "cifar10.csv",10)

        self.train_loader = []
        for client_id in range(self.num_parts):
            indices = data_subsets[client_id]
            subset_data = Subset(self.training_data, indices)
            dataloader = DataLoader(subset_data, batch_size=self.train_batch_size, shuffle=False)
            self.train_loader.append(dataloader)

        self.test_data = datasets.CIFAR10(
            root=root_path,
            train=False,
            download=True,
            transform=self.transform
        )

        if test_size != 0:
            subset_test_data = Subset(self.test_data, list(range(test_size)))
            self.test_loader = DataLoader(subset_test_data, batch_size=self.test_batch_size, shuffle=False, pin_memory=True)
        else:
            self.test_loader = DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=False, pin_memory=True)


class MNIST:
    def __init__(self, config):
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.num_parts = config["num_parts_dataset"]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.device = torch.device(config["device"])

        seed = config["seed"]
        alpha = config["alpha"]
        iid_ratio = config["iid_ratio"]
        test_size = config["test_size"]
        logger.debug(f"Selected alpha for dirichlet distribution: {alpha}, selected iid ratio: {iid_ratio}")

        self.training_data = datasets.MNIST(
            root=root_path,
            train=True,
            download=True,
            transform=self.transform
        )
        if alpha != 1:
            data_subsets = MNISTPartitioner(self.training_data.targets,
                                            num_clients=self.num_parts,
                                            partition="noniid-labeldir",
                                            dir_alpha=alpha,
                                            seed=seed)
        elif iid_ratio != 1:
            data_subsets = MNISTPartitioner(self.training_data.targets,
                                            num_clients=self.num_parts,
                                            partition="noniid-#label",
                                            major_classes_num=int(iid_ratio * 10),
                                            seed=seed)
        else:
            data_subsets = MNISTPartitioner(self.training_data.targets,
                                            num_clients=self.num_parts,
                                            partition="iid",
                                            seed=seed)

        # plot graphs about distribution of the data
        # plot_graph(self.training_data.targets, data_subsets.client_dict, "mnist.csv", 10)

        # each client as a different dataloader (selected with client index)
        self.train_loader = []
        for client_id in range(self.num_parts):
            indices = data_subsets[client_id]
            subset_data = Subset(self.training_data, indices)
            dataloader = DataLoader(subset_data, batch_size=self.train_batch_size, shuffle=False)
            self.train_loader.append(dataloader)

        # test data not divided in subset
        self.test_data = datasets.MNIST(
            root="./Data",
            train=False,
            download=True,
            transform=self.transform
        )

        if test_size != 0:
            subset_test_data = Subset(self.test_data, list(range(test_size)))
            self.test_loader = DataLoader(subset_test_data, batch_size=self.test_batch_size, shuffle=False, pin_memory=True)
        else:
            self.test_loader = DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=False, pin_memory=True)

class MedMNIST:
    def __init__(self, config):
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.num_parts = config["num_parts_dataset"]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.device = torch.device(config["device"])

        seed = config["seed"]
        alpha = config["alpha"]
        iid_ratio = config["iid_ratio"]
        test_size = config["test_size"]
        logger.debug(f"Selected alpha for dirichlet distribution: {alpha}, selected iid ratio: {iid_ratio}")

        self.training_data = PathMNIST(
            split='train',
            root=root_path,
            download=True,
            transform=self.transform,
            as_rgb=True
        )

        array = np.array(self.training_data.labels)
        targets = torch.from_numpy(array).squeeze()

        if alpha != 1:
            data_subsets = MNISTPartitioner(targets,
                                            num_clients=self.num_parts,
                                            partition="noniid-labeldir",
                                            dir_alpha=alpha,
                                            seed=seed)
        else:
            data_subsets = MNISTPartitioner(targets,
                                            num_clients=self.num_parts,
                                            partition="iid",
                                            seed=seed)

        
        # plot_graph(targets, data_subsets.client_dict, "medmnist.csv", 9)

        self.train_loader = []
        for client_id in range(self.num_parts):
            indices = data_subsets[client_id]
            subset_data = Subset(self.training_data, indices)
            dataloader = DataLoader(subset_data, batch_size=self.train_batch_size, shuffle=False)
            self.train_loader.append(dataloader)

        # Carica i dati di test
        self.test_data = PathMNIST(
            split='test',
            root=root_path,
            download=True,
            transform=self.transform,
            as_rgb=True
        )

        if test_size != 0:
            subset_test_data = Subset(self.test_data, list(range(test_size)))
            self.test_loader = DataLoader(subset_test_data, batch_size=self.test_batch_size, shuffle=False, pin_memory=True)
        else:
            self.test_loader = DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)


class MedMNIST2: 
    def __init__(self, config):
        self.train_batch_size = config["train_batch_size"]
        self.test_batch_size = config["test_batch_size"]
        self.num_parts = config["num_parts_dataset"]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        self.device = torch.device(config["device"])
        
        load_dataset = config["load_dataset"]
        seed = config["seed"]
        alpha = config["alpha"]
        iid_ratio = config["iid_ratio"]
        test_size = config["test_size"]

        logger.debug(f"Selected alpha for dirichlet distribution: {alpha}, selected iid ratio: {iid_ratio}")

        self.training_data = PathMNIST(
            split='train',
            root=root_path,
            download=True,
            transform=self.transform,
            as_rgb=True
        )

        array = np.array(self.training_data.labels)
        targets = torch.from_numpy(array).squeeze()

        partition_file = f"partition_alpha{alpha}_numParts{self.num_parts}.pkl"

        # === STEP 1: Load or Generate client_dict ===
        if load_dataset and os.path.exists(partition_file):
            logger.info(f"Loading existing dataset partition from {partition_file}")
            client_dict = torch.load(partition_file)
        
        else:
            logger.info(f"Generating new dataset partition and saving to {partition_file}")
            if alpha != 1:
                data_subsets = MNISTPartitioner(
                    targets,
                    num_clients=self.num_parts,
                    partition="noniid-labeldir",
                    dir_alpha=alpha,
                    seed=seed
                )
            else:
                data_subsets = MNISTPartitioner(
                    targets,
                    num_clients=self.num_parts,
                    partition="iid",
                    seed=seed
                )
            client_dict = data_subsets.client_dict # Create a dictionary with the indices of the data for each client
            torch.save(client_dict, partition_file)
            
       # === STEP 2: Create DataLoaders using loaded or generated client_dict ===
        self.train_loader = []
        for client_id in range(self.num_parts):
            indices = client_dict[client_id]
            subset_data = Subset(self.training_data, indices)
            dataloader = DataLoader(subset_data, batch_size=self.train_batch_size, shuffle=False)
            self.train_loader.append(dataloader)    
        
        
        # === STEP 3: Create Test Loader ===
        self.test_data = PathMNIST(
            split='test',
            root=root_path,
            download=True,
            transform=self.transform,
            as_rgb=True
        )

        if test_size != 0:
            subset_test_data = Subset(self.test_data, list(range(test_size)))
            self.test_loader = DataLoader(subset_test_data, batch_size=self.test_batch_size, shuffle=False, pin_memory=True)
        else:
            self.test_loader = DataLoader(self.test_data, batch_size=self.test_batch_size, shuffle=False, num_workers=4, pin_memory=True)
        


# here you can create import other datasets
def plot_graph(targets, client_dict, csv_file,class_num):
    partition_report(targets, client_dict,
                     class_num=class_num, verbose=False, file=csv_file)

    partition_df = pd.read_csv(csv_file, header=1)
    partition_df = partition_df.set_index('client')
    col_names = [f"class{i}" for i in range(class_num)]
    for col in col_names:
        partition_df[col] = (partition_df[col] * partition_df['Amount']).astype(int)

    partition_df[col_names].iloc[:10].plot.barh(stacked=True)
    plt.tight_layout()
    plt.xlabel('sample num')
    plt.savefig(f"data_distribution.png",
                dpi=200, bbox_inches='tight')
