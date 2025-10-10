import csv
import matplotlib.pyplot as plt


def plot_accuracy_from_file(filename):
    epochs = []
    accuracies = []

    with open(filename, mode='r') as file:
        reader = csv.reader(file)
        next(reader)  
        for row in reader:
            epochs.append(int(row[0]))
            accuracies.append(float(row[1]))

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, accuracies, linestyle='-', color='b', label="Test Accuracy", linewidth=1)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy over Epochs")
    plt.legend()
    plt.grid(True)
    plt.show()


plot_accuracy_from_file('accuracy.csv')
