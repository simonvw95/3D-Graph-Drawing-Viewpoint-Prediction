import torch
import torch.nn as nn
from viewpoint_optimization.metrics_gd_torch import cross_pairs
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm


# taken from https://github.com/tiga1231/graph-drawing/blob/sgd/neural-crossing-detector.ipynb
# class for creating a random edge pair dataset
class EdgePairDataset:
    def __init__(self, n=10000):
        super().__init__()
        self.n = n
        self.data = torch.rand(n, 8)
        self.label = cross_pairs(self.data)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.data[i], self.label[i]


# simple Multilayer Perceptron
class CrossingDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_dims = [8, 96, 256, 96, 1]
        self.layers = []
        for i, (in_dim, out_dim) in enumerate(zip(self.layer_dims[:-1], self.layer_dims[1:])):
            self.layers.append(nn.Linear(in_dim, out_dim))
            if i < len(self.layer_dims) - 2:
                self.layers.append(nn.LeakyReLU())
                self.layers.append(nn.LayerNorm(out_dim))
            else:
                self.layers.append(nn.Sigmoid())
        self.main = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.main(x)


def train_mlp(lr = 0.05, momentum = 0.9, epochs = 20, batch_size = 1024, dataset_size = 1e6, device = 'cuda'):

    dataset = EdgePairDataset(n = int(dataset_size))
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    device = device
    model = CrossingDetector().to(device)
    bce = nn.BCELoss()
    # grid search experiment was done to determine the best lr and momentum
    # a high value of momentum > 0.7 and a lr between 0.02 and 0.08 showed consistent accuracies above 98%
    optmizer = optim.SGD(model.parameters(), lr = lr, momentum = momentum)
    loss_curve = []

    # training
    for ep in tqdm(range(epochs)):
        for edge_pairs, targets in dataloader:
            edge_pairs, targets = edge_pairs.to(device), targets.to(device)
            pred = model(edge_pairs)
            loss = bce(pred, targets.float().view(-1, 1))

            optmizer.zero_grad()
            loss.backward()
            optmizer.step()

        loss_curve.append(loss.item())

    # 98.495% accuracy, 1.5% absolute increase over the one described in Neural Aesthete paper
    torch.save(model.state_dict(), 'evaluations/mlp_cross.pt')

    # testing saved model
    test_loader = DataLoader(EdgePairDataset(n = int(1e5)), batch_size = batch_size, shuffle = True)

    correct = 0
    total = 0
    with torch.no_grad():
        for edge_pairs, targets in tqdm(test_loader):
            edge_pairs, targets = edge_pairs.to(device), targets.to(device)
            pred = model(edge_pairs)
            correct += ((pred > 0.5) == targets.view(-1, 1)).sum().item()
            total += len(targets)

    print(f'{correct}/{total} {correct / total}')
