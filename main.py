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

def tune_lr():
    data_dir = os.path.abspath("./data")
    config = {
        'l1': 256,
        'l2': 8,
        'lr': tune.loguniform(1e-4, 1e-1),
        'batch_size': 4}
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=5,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        partial(train_tune, data_dir=data_dir),
        resources_per_trial={"cpu": 12, "gpu": 1},
        local_dir="./ray_results",
        config=config,
        num_samples=5,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    best_checkpoint = result.get_best_checkpoint(best_trial, metric="loss", mode="min", )
def train_model(epochs, config, from_checkpoint=False, path=None):
    data_dir = os.path.abspath("./data")
    train(epochs, config, data_dir, from_checkpoint, path)

def main():
    config = {'l1': 256, 'l2': 8, 'lr': 0.0008611640946726886, 'batch_size': 4}
    train_model(25, config)
    #tune_lr()

if __name__ == "__main__":
    main()