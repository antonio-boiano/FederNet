# extra, used only to test prune network routine
import time
import torch
from FedOpt.src.Optimizations.ModelPruning.prune import prune_network
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict

def complete_test_pruning(model,test_loader):
    prune_step_ratio = 1 / 8
    max_channel_ratio = 0.90

    prune_channels = [64, 64, 128, 128, 256, 256, 256, 512, 512, 512, 512, 512, 512]
    prune_layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8', 'conv9', 'conv10',
                    'conv11', 'conv12', 'conv13']

    top1_accuracies = {}
    top5_accuracies = {}

    for conv, channel in zip(prune_layers, prune_channels):
        top1_accuracies[conv] = []
        top5_accuracies[conv] = []

        # load new network and check accuracy
        network, _, _ = test_network(model, test_loader)

        # remove 0 channels ~ M (max_channel_ratio) % of total channels
        step = np.linspace(0, int(channel * max_channel_ratio), int(1 / prune_step_ratio), dtype=int)
        steps = (step[1:] - step[:-1]).tolist()

        for i in range(len(steps)):
            print("\n%s: %s Layer, %d Channels pruned" % (time.ctime(), conv, sum(steps[:i + 1])))

            network = prune_network(network,[conv],[steps[i]])

            network, _, (top1, top5) = test_network(network, test_loader)

            top1_accuracies[conv].append(top1)
            top5_accuracies[conv].append(top5)

    plot_accuracy_graph(top1_accuracies,"Top1 Accuracy")
    plot_accuracy_graph(top5_accuracies,"Top5 Accuracy")

def test_network(network=None, data_loader=None):
    device = torch.device("cpu")
    top1, top5 = test_step(network, data_loader, device)
    return network, None, (top1, top5)

def test_step(network, data_loader, device):
    network.eval()

    data_time = AverageMeter()
    forward_time = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    with torch.no_grad():
        tic = time.time()
        for inputs, targets in data_loader:
            data_time.update(time.time() - tic)

            inputs, targets = inputs.to(device), targets.to(device)

            tic = time.time()
            outputs = network(inputs)
            forward_time.update(time.time() - tic)

            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))

            top1.update(prec1.item(), inputs.size(0))
            top5.update(prec5.item(), inputs.size(0))

            tic = time.time()

    str_ = '%s: Test information, ' % time.ctime()
    str_ += 'Data(s): %2.3f, Forward(s): %2.3f, ' % (data_time.sum, forward_time.sum)
    str_ += 'Top1: %2.3f, Top5: %2.3f, ' % (top1.avg, top5.avg)
    print("-*-" * 10 + "\n\tEvalute network\n" + "-*-" * 10)
    print(str_)

    return top1.avg, top5.avg


def accuracy(output, target, topk=(1,)):
    """
        Computes the precision@k for the specified values of k
        ref: https://github.com/chengyangfu/pytorch-vgg-cifar10
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def plot_accuracy_graph(data: Dict,text="Accuracy"):
    # plot
    colors = ['r', 'g', 'b', 'k', 'y', 'm', 'c']
    lines = ['-', '--', '-.']

    plt.figure(figsize=(7, 5))
    for index, (key, value) in enumerate(data.items()):
        line_style = colors[index % len(colors)] + lines[index // len(colors)] + 'o'
        plt.plot(np.linspace(0, 95, len(value)), value, line_style, label=key)

    plt.title("Pruned smallest filters")
    plt.ylabel(text)
    plt.xlabel("Filters Pruned Away (%)")
    plt.legend(loc='lower left')
    plt.grid()
    plt.xlim(0, 100)
    #plt.savefig("figure.png", dpi=150, bbox_inches='tight')
    plt.show()

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count