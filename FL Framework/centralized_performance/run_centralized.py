import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from medmnist import PathMNIST
import numpy as np
from fedlab.utils.dataset import MNISTPartitioner
import os


class LeNetPathMNISTModified(nn.Module):
    def __init__(self, num_classes=9):
        super(LeNetPathMNISTModified, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        feature_size = 32 * 1 * 1
        self.classifier = nn.Sequential(
            nn.Linear(feature_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def train(model, train_loader, optimizer, criterion, device):
    model.train()
    for images, labels in train_loader:
        labels = labels.squeeze().long()
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            labels = labels.squeeze().long()
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    accuracy = correct / total
    return accuracy


# def create_dataloaders(config):
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
#     root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './Data'))    
#     device = torch.device(config["device"])

#     full_train_dataset = PathMNIST(
#         split='train',
#         root=root_path,
#         download=True,
#         transform=transform,
#         as_rgb=True
#     )

#     full_test_dataset = PathMNIST(
#         split='test',
#         root=root_path,
#         download=True,
#         transform=transform,
#         as_rgb=True
#     )

#     labels_array = np.array(full_train_dataset.labels).squeeze()
#     targets = torch.from_numpy(labels_array)

#     if config["alpha"] != 1:
#         partitioner = MNISTPartitioner(
#             targets,
#             num_clients=config["num_parts_dataset"],
#             partition="noniid-labeldir",
#             dir_alpha=config["alpha"],
#             seed=config["seed"]
#         )
#     else:
#         partitioner = MNISTPartitioner(
#             targets,
#             num_clients=config["num_parts_dataset"],
#             partition="iid",
#             seed=config["seed"]
#         )

#     all_indices = []
#     for client_id in range(config["num_parts_dataset"]):
#         all_indices.extend(partitioner[client_id])

#     centralized_train_dataset = Subset(full_train_dataset, all_indices)
#     train_loader = DataLoader(centralized_train_dataset, batch_size=config["train_batch_size"], shuffle=True)

#     if config["test_size"] > 0:
#         subset_test_dataset = Subset(full_test_dataset, list(range(config["test_size"])))
#         test_loader = DataLoader(subset_test_dataset, batch_size=config["test_batch_size"], shuffle=False, pin_memory=True)
#     else:
#         test_loader = DataLoader(full_test_dataset, batch_size=config["test_batch_size"], shuffle=False, num_workers=4, pin_memory=True)

#     return train_loader, test_loader


# def main():
#     config = {
#         "train_batch_size": 300,
#         "test_batch_size": 300,
#         "num_parts_dataset": 10,
#         "device": "cuda" if torch.cuda.is_available() else "cpu",
#         "seed": 42,
#         "alpha": 10,
#         "iid_ratio": 1.0,
#         "test_size": 0, 
#         "epochs": 100
#     }

#     epochs = config["epochs"]
#     device = torch.device(config["device"])
    
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))
#     ])
        
#     # === Load centralized datasets ===
#     # train_dataset = PathMNIST(split='train', download=True, transform=transform, as_rgb=True)
#     # test_dataset = PathMNIST(split='test', download=True, transform=transform, as_rgb=True)

#     root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './Data'))    

#     train_dataset = PathMNIST(
#         split='train',
#         root=root_path,
#         download=True,
#         transform=transform,
#         as_rgb=True
#     )

#     test_dataset = PathMNIST(
#         split='test',
#         root=root_path,
#         download=True,
#         transform=transform,
#         as_rgb=True
#     )
    
#     train_loader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False, num_workers=4, pin_memory=True)

#     model = LeNetPathMNISTModified().to(device)
#     optimizer = optim.Adam(model.parameters(), lr=0.001)
#     criterion = nn.CrossEntropyLoss()

#     for epoch in range(epochs):
#         train(model, train_loader, optimizer, criterion, device)
#         acc = evaluate(model, test_loader, device)
#         print(f"Epoch {epoch+1}, Accuracy: {acc:.4f}")


# if __name__ == "__main__":
#     main()
import io

def get_model_size_mb(model: nn.Module) -> float:
    """Calculate the size of a PyTorch model in megabytes."""
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    size_mb = buffer.getbuffer().nbytes / (1024 ** 2)
    return size_mb


def main():
    config = {
        "train_batch_size": 300,
        "test_batch_size": 300,
        "num_parts_dataset": 10,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "seed": 42,
        "alpha": 10,
        "iid_ratio": 1.0,
        "test_size": 0, 
        "epochs": 100
    }

    epochs = config["epochs"]
    device = torch.device(config["device"])
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
        
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), './Data'))    

    train_dataset = PathMNIST(
        split='train',
        root=root_path,
        download=True,
        transform=transform,
        as_rgb=True
    )

    test_dataset = PathMNIST(
        split='test',
        root=root_path,
        download=True,
        transform=transform,
        as_rgb=True
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config["train_batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["test_batch_size"], shuffle=False, num_workers=4, pin_memory=True)

    model = LeNetPathMNISTModified().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    size_mb = get_model_size_mb(model)
    print(f"Model size: {size_mb:.2f} MB")

    for epoch in range(epochs):
        train(model, train_loader, optimizer, criterion, device)
        acc = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1}, Accuracy: {acc:.4f}")
        test_accuracy = []
        test_accuracy.append(acc)

    # Plotting the test accuracy over epochs
    # import matplotlib.pyplot as plt   
    # plt.plot(range(1, epochs + 1), test_accuracy)
    # plt.xlabel('Epochs')
    # plt.ylabel('Test Accuracy')
    # plt.title('Test Accuracy over Epochs')
    # plt.show()
    
    # Save test accuracy to a file
    with open('test_accuracy.txt', 'w') as f:
        for epoch, acc in enumerate(test_accuracy, start=1):
            f.write(f"Epoch {epoch}: {acc:.4f}\n")

if __name__ == "__main__":
    main()
    
    