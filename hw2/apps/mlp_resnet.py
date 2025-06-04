import sys

import needle as ndl
import needle.nn as nn
import numpy as np
import time
from needle.data import MNISTDataset, DataLoader

np.random.seed(0)
# MY_DEVICE = ndl.backend_selection.cuda()


def ResidualBlock(dim, hidden_dim, norm=nn.BatchNorm1d, drop_prob=0.1):
    seq = nn.Sequential(nn.Linear(dim, hidden_dim), 
                        norm(hidden_dim), nn.ReLU(), nn.Dropout(drop_prob), 
                        nn.Linear(hidden_dim, dim), norm(dim))
    return nn.Sequential(nn.Residual(seq), nn.ReLU())


def MLPResNet(
    dim,
    hidden_dim=100,
    num_blocks=3,
    num_classes=10,
    norm=nn.BatchNorm1d,
    drop_prob=0.1,
):
    return nn.Sequential(nn.Flatten(), nn.Linear(dim, hidden_dim), nn.ReLU(),
                        *[ResidualBlock(hidden_dim, hidden_dim//2, norm, drop_prob) for _ in range(num_blocks)],
                        nn.Linear(hidden_dim, num_classes))


def epoch(dataloader, model, opt=None):
    np.random.seed(4)
    
    if opt is not None:
        model.train()
    else:
        model.eval()
    
    loss_fn = nn.SoftmaxLoss()
    total_loss = 0.0
    total_errors = 0
    total_samples = 0
    
    for batch_idx, (X, y) in enumerate(dataloader):
        X = ndl.Tensor(X, dtype="float32")
        y = ndl.Tensor(y, dtype="int32")
        
        logits = model(X)
        loss = loss_fn(logits, y)
        
        predicted = np.argmax(logits.numpy(), axis=1)
        errors = np.sum(predicted != y.numpy())
        
        total_loss += loss.numpy() * X.shape[0]
        total_errors += errors
        total_samples += X.shape[0]
        
        if opt is not None:
            opt.reset_grad()
            loss.backward()
            opt.step()
    avg_error_rate = total_errors / total_samples
    avg_loss = total_loss / total_samples
    
    return avg_error_rate, avg_loss


def train_mnist(
    batch_size=100,
    epochs=10,
    optimizer=ndl.optim.Adam,
    lr=0.001,
    weight_decay=0.001,
    hidden_dim=100,
    data_dir="data",
):
    np.random.seed(4)
    resnet = MLPResNet(28*28, hidden_dim=hidden_dim, num_classes=10)
    opt = optimizer(resnet.parameters(), lr=lr, weight_decay=weight_decay)
    train_set = MNISTDataset(f"{data_dir}/train-images-idx3-ubyte.gz", 
                             f"{data_dir}/train-labels-idx1-ubyte.gz")
    test_set = MNISTDataset(f"{data_dir}/t10k-images-idx3-ubyte.gz",
                            f"{data_dir}/t10k-labels-idx1-ubyte.gz")
    train_load = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_load = DataLoader(test_set, batch_size=batch_size)
    for _ in range(epochs):
        train_err, train_loss = epoch(train_load, resnet, opt)
    test_err, test_loss = epoch(test_load, resnet, None)
    return train_err, train_loss, test_err, test_loss


if __name__ == "__main__":
    train_mnist(data_dir="../data")
