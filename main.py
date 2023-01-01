from train import *
from model import *

import torch
import torch.nn as nn

import numpy as np
from functools import partial
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler


def tune_hyperparameters():
    data_dir = os.path.abspath("./data")
    config = {
        "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([2, 4, 8, 16])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=4,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train_tune, data_dir=data_dir),
        resources_per_trial={"cpu": 12, "gpu": 1},

        config=config,
        num_samples=10,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
def train_model(epochs, config):
    data_dir = os.path.abspath("./data")
    train(epochs, config, data_dir)

def main():
    config = {'l1': 256, 'l2': 8, 'lr': 0.05557941010651975, 'batch_size': 4}
    train_model(25)

if __name__ == "__main__":
    main()