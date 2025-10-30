import pprint
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dotenv import load_dotenv
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from wandb.sklearn import (
    plot_calibration_curve,
    plot_class_proportions,
    plot_clusterer,
    plot_confusion_matrix,
    plot_elbow_curve,
    plot_feature_importances,
    plot_learning_curve,
    plot_outlier_candidates,
    plot_precision_recall,
    plot_residuals,
    plot_regressor,
    plot_roc,
    plot_silhouette,
    plot_summary_metrics,
)

import wandb

load_dotenv()


def train():
    config = {
        "learning_rate": 0.02,
        "architecture": "CNN",
        "dataset": "CIFAR-100",
        "epochs": 10,
    }
    run = wandb.init(
        project="ml-training",
        config=config,
    )

    # Simulate training.
    epochs = 10
    offset = random.random() / 5
    for epoch in range(2, epochs):
        acc = 1 - 2**-epoch - random.random() / epoch - offset
        loss = 2**-epoch + random.random() / epoch + offset

        metrics = {"acc": acc, "loss": loss}
        run.log(metrics)
    run.finish()


# load and process data
def sklearn_train():
    wbcd = datasets.load_breast_cancer()
    feature_names = wbcd.feature_names
    labels = wbcd.target_names

    test_size = 0.2
    X_train, X_test, y_train, y_test = train_test_split(
        wbcd.data,
        wbcd.target,
        test_size=test_size,
    )

    # train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    model_params = model.get_params()

    # get predictions
    y_pred = model.predict(X_test)
    y_probas = model.predict_proba(X_test)
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # start a new wandb run and add your model hyperparameters
    wandb.init(
        project="ml-training",
        config=model_params,
    )

    # Add additional configs to wandb
    update_config = {
        "test_size": test_size,
        "train_len": len(X_train),
        "test_len": len(X_test),
    }
    wandb.config.update(update_config)

    plot_class_proportions(y_train, y_test, labels)
    plot_learning_curve(model, X_train, y_train)
    plot_roc(y_test, y_probas, labels)
    plot_precision_recall(y_test, y_probas, labels)
    plot_feature_importances(model)

    wandb.finish()
    # or

    # wandb.init(project="visualize-sklearn") as run:
    #     y_pred = clf.predict(X_test)
    #     accuracy = sklearn.metrics.accuracy_score(y_true, y_pred)
    #
    #     # If logging metrics over time, then use run.log
    #     run.log({"accuracy": accuracy})
    #
    #     # OR to log a final metric at the end of training you can also use run.summary
    #     run.summary["accuracy"] = accuracy


def model_sweeps():

    sweep_config = build_sweet_config()

    pprint.pprint(sweep_config)

    sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train_pytorch(config=None):
        # Initialize a new wandb run
        with wandb.init(config=config) as run:
            # If called by wandb.agent, as below,
            # this config will be set by Sweep Controller
            config = run.config

            loader = build_dataset(config.batch_size)
            network = build_network(config.fc_layer_size, config.dropout)
            optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

            for epoch in range(config.epochs):
                avg_loss = train_epoch(network, loader, optimizer)
                run.log({"loss": avg_loss, "epoch": epoch})

    def build_dataset(batch_size):

        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        # download MNIST training dataset
        dataset = datasets.MNIST(".", train=True, download=True, transform=transform)
        sub_dataset = torch.utils.data.Subset(
            dataset, indices=range(0, len(dataset), 5)
        )
        loader = torch.utils.data.DataLoader(sub_dataset, batch_size=batch_size)

        return loader

    def build_network(fc_layer_size, dropout):
        network = nn.Sequential(  # fully connected, single hidden layer
            nn.Flatten(),
            nn.Linear(784, fc_layer_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_layer_size, 10),
            nn.LogSoftmax(dim=1),
        )

        return network.to(device)

    def build_optimizer(network, optimizer, learning_rate):
        if optimizer == "sgd":
            optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
        elif optimizer == "adam":
            optimizer = optim.Adam(network.parameters(), lr=learning_rate)
        return optimizer

    def train_epoch(network, loader, optimizer):
        cumu_loss = 0

        with wandb.init() as run:
            for _, (data, target) in enumerate(loader):
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()

                # ➡ Forward pass
                loss = F.nll_loss(network(data), target)
                cumu_loss += loss.item()

                # ⬅ Backward pass + weight update
                loss.backward()
                optimizer.step()
                batch_loss = loss.item()
                run.log({"batch loss": batch_loss})

        return cumu_loss / len(loader)


def build_sweet_config() -> dict[
    str,
    str
    | dict[str, str]
    | dict[
        str,
        dict[str, list[str]]
        | dict[str, list[int]]
        | dict[str, list[float]]
        | dict[str, str | int | float]
        | dict[str, str | int]
        | dict[str, int],
    ],
]:
    parameters = {
        "optimizer": {"values": ["adam", "sgd"]},
        "fc_layer_size": {"values": [128, 256, 512]},
        "dropout": {"values": [0.3, 0.4, 0.5]},
        "learning_rate": {
            # a flat distribution between 0 and 0.1
            "distribution": "uniform",
            "min": 0,
            "max": 0.1,
        },
        "batch_size": {
            # integers between 32 and 256
            # with evenly-distributed logarithms
            "distribution": "q_log_uniform_values",
            "q": 8,
            "min": 32,
            "max": 256,
        },
        "epochs": {"value": 1},
    }

    metric = {
        "name": "loss",
        "goal": "minimize",
    }
    sweep_config = {
        "method": "random",
        "metric": metric,
        "parameters": parameters,
    }
    return sweep_config


def viz():
    _ = [
        plot_calibration_curve(),
        plot_clusterer(),
        plot_confusion_matrix(),
        plot_elbow_curve(),
        plot_outlier_candidates(),
        plot_residuals(),
        plot_regressor(),
        plot_silhouette(),
        plot_summary_metrics(),
    ]


if __name__ == "__main__":
    train()
    pass
