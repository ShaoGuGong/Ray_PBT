import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

import ray
from ray import tune
from ray.util.placement_group import (
    placement_group,
    placement_group_table,
    remove_placement_group,
)
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from numpy import random
import time
import copy
import os
import heapq
import json
import math
import matplotlib.pyplot as plt
import argparse


HEAD_NODE_IP = "192.168.50.35"  # Head node IP
HYPER_NUM = 20  # Number of hyperparameters
BATCH_SIZE = 512  # Batch size for one training iteration
STOP_ITER = 1000  # Total number of training iterations
STOP_ACC = 0.8  # Stop training when accuracy reaches this value
INTERVAL_REPORT = 300  # Interval for displaying progress in the terminal (in seconds)
INTERVAL_CHECK = 50
RESOURCE_ALLOCATION = {  # Resource allocation for each computer
    "CPU": {
        "192.168.50.34": 16,
        "192.168.50.35": 12,
        "192.168.50.36": 4,
        "192.168.50.37": 4,
        "192.168.50.38": 4,
        "192.168.50.8": 20,
        "192.168.50.19": 12,
        "192.168.50.88": 12,
    },
    "GPU": {"192.168.50.35": 1},
}
# TEST_SIZE = 25


# Create data loaders
def get_data_loader(
    model_type,
    batch_size=64,
    data_dir="/home/ray_cluster/Documents/workspace/tune_population_based/data",
):
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    if model_type == "resnet-18":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(
                root=data_dir, train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=False,
        )
    elif model_type == "resnet-50":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=True, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=True,
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(
                root=data_dir, train=False, download=True, transform=transform
            ),
            batch_size=batch_size,
            shuffle=False,
        )
    return train_loader, test_loader


# Model training
def train(model, optimizer, train_loader, device=None):
    model.train()
    criterion = nn.CrossEntropyLoss().to(device)
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        break


def test(model, test_loader, device):
    total = 0
    correct = 0
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    return correct / total


# Used to display the status of all trials (displayed in the terminal)
@ray.remote(num_cpus=0.1, resources={"node:" + HEAD_NODE_IP: 0.1})
def Reporter(tuner, max_report_frequency=5, hyper_num=1):
    start_time = ray.get(tuner.get_start_time.remote())
    resource = ray.get(tuner.get_resource.remote())
    while True:
        hypers, accuracy, state, perturbs, running_trial_num, package_size = ray.get(
            tuner.get_for_reporter.remote()
        )
        m, s = divmod(time.time() - start_time, 60)
        h, m = divmod(m, 60)
        if "CPU" in ray.available_resources():
            unused_cpu_num = ray.available_resources()["CPU"]
        else:
            unused_cpu_num = 0
        if "GPU" in ray.available_resources():
            unused_gpu_num = ray.available_resources()["GPU"]
        else:
            unused_gpu_num = 0

        print("== Status ==")
        print(
            f"Current Time : {time.ctime()} (running for {str(int(h)).zfill(2)}:{str(int(m)).zfill(2)}:{str(int(s)).zfill(2)})"
        )
        print(f"Unused Resource : {unused_cpu_num} CPUs and {unused_gpu_num} GPUs")
        print(f"PBT : {perturbs} perturbs")
        print(
            f"Total hypers : {hyper_num} ( {running_trial_num} is training ), package_size : {package_size}"
        )
        print(
            "+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+"
        )
        print(
            "| Hyper name   |   Status   |  CPU / GPU |                  IP |         lr |   momentum | batch_size |      acc |  iter |   total time (s)|"
        )
        print(
            "+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+"
        )
        for i, (hyper, acc, sta) in enumerate(zip(hypers, accuracy, state)):
            if sta["resource_id"] == -2:
                status = "TERMINAL"
                cpus_per_trial = 0
                gpus_per_trial = 0
                ip = "node:0.0.0.0"
            elif sta["resource_id"] == -1:
                status = "PENDING"
                cpus_per_trial = 0
                gpus_per_trial = 0
                ip = "node:0.0.0.0"
            else:
                status = "RUNNING"
                cpus_per_trial = resource[sta["resource_id"]]["CPU"]
                gpus_per_trial = resource[sta["resource_id"]]["GPU"]
                ip = resource[sta["resource_id"]]["node"]

            print(
                f"| hyper_{str(i).zfill(5)}  |  {status:^8}  | {cpus_per_trial:>4.1f} / {gpus_per_trial:<3.1f} | {ip:>19} | {hyper['lr']:10.6f} | {hyper['momentum']:10.6f} | {hyper['batch_size']:>11}| {acc:8.4f} | {sta['iteration']:>5} | {sta['run_time']:15.6f} | "
            )
        print(
            "+--------------+------------+------------+---------------------+------------+------------+------------+----------+-------+-----------------+"
        )
        time.sleep(max_report_frequency)


@ray.remote(num_cpus=0.2, resources={"node:" + HEAD_NODE_IP: 0.1})
class Tuner(object):
    """
    Tuner: Controls all trial processes, creates and receives trial results.

    Args:
        hyper_num: Number of hyperparameter configurations to create
        batch_size: Batch size for one training iteration
        stop_acc: Accuracy condition to stop training
        stop_iteration: Iteration condition to stop training
        checkpoint_interval: Number of iterations between checkpoints
        trials_state: Stores the resource ID used by each hyperparameter training and the time spent on training
        resource: Stores the total allocated resources based on RESOURCE_ALLOCATION
        available_resource: Stores currently available resources (excluding those in use or with poor performance)
        trials_scheduler: Stores hyper_ids to be trained (removed once termination conditions are met)
        running_trial_num: Number of trials currently being trained
        min_run_one_interval_time: Minimum time required to run one interval (basis for calculating resource capabilities)
    """

    def __init__(
        self,
        hyper_num=1,
        model_type="resnet-18",
        resource_allocation=None,
        stop_acc=1,
        stop_iteration=0,
        checkpoint_interval=5,
        hyperparam_mutations=None,
        path=None,
    ):
        self.start_time = time.time()
        self.tuner = None
        self.hyper_num = hyper_num
        self.model_type = model_type
        self.stop_acc = stop_acc
        self.stop_iteration = stop_iteration
        self.checkpoint_interval = checkpoint_interval
        self.hyperparam_mutations = hyperparam_mutations
        self.path = path

        self.trials_scheduler = []
        self.hypers = []
        self.trials_state = []
        self.checkpoints = []
        self.last_checkpoint = [0] * hyper_num
        self.perturbs = 0
        self.trial_acc_list = [0] * hyper_num
        self.resource = []
        self.avaliable_resource = []

        self.running_trial_num = 0
        self.running_resource_num = 0
        self.min_run_one_interval_time = 9999
        self.max_iter = 0
        self.max_acc = -1
        self.last_run_interval = 9999
        self.package_size = 0

        self.initialize_all_config()
        self.set_placement_group(resource_allocation)

    # Initialize each hyperparameter configuration and checkpoint
    def initialize_all_config(self):
        if self.model_type == "resnet-18":
            # Create model
            model = models.resnet18()
            # Modify the output of the fully connected layer
            model.fc = nn.Linear(model.fc.in_features, 10)
        elif self.model_type == "resnet-50":
            # Create model
            model = models.resnet50()
            # Modify the output of the fully connected layer
            model.fc = nn.Linear(model.fc.in_features, 100)
        optimizer = optim.SGD(
            model.parameters(),
            lr=0.1,
            momentum=0.9,
        )

        for i in range(self.hyper_num):
            hyper = {
                "lr": random.uniform(0.001, 1),
                "momentum": random.uniform(0.001, 1),
                "batch_size": 512,
                "model_type": self.model_type,
            }
            trial_state = {
                "resource_id": -1,
                "run_time": 0,
                "iteration": 0,
            }
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "checkpoint_interval": self.checkpoint_interval,
            }
            self.trials_scheduler.append(i)
            self.hypers.append(hyper)
            self.trials_state.append(trial_state)
            self.checkpoints.append(checkpoint)

    # Allocate the number of resources each node can use during training
    def set_placement_group(self, resource_allocation):
        # print(ray.available_resources())
        for nodes in ray.nodes():
            if (
                "CPU" in nodes["Resources"]
                and nodes["NodeManagerAddress"] in resource_allocation["CPU"]
            ):
                if nodes["NodeManagerAddress"] == HEAD_NODE_IP:
                    sub = 1
                else:
                    sub = 0
                sum = nodes["Resources"]["CPU"]
                while int(
                    sum / (resource_allocation["CPU"][nodes["NodeManagerAddress"]])
                ):
                    self.resource.append(
                        {
                            "CPU": resource_allocation["CPU"][
                                nodes["NodeManagerAddress"]
                            ]
                            - sub,
                            "GPU": 0,
                            "node": "node:" + nodes["NodeManagerAddress"],
                            "calculate_ability": 0,
                            "Used_count": 0.0,
                        }
                    )
                    print(self.resource[-1])
                    sub = 0
                    sum -= resource_allocation["CPU"][nodes["NodeManagerAddress"]]
                    # print("CPU", resource_allocation["CPU"][nodes['NodeManagerAddress']])
            if (
                "GPU" in nodes["Resources"]
                and nodes["NodeManagerAddress"] in resource_allocation["GPU"]
            ):
                sum = nodes["Resources"]["GPU"]
                while int(
                    sum / resource_allocation["GPU"][nodes["NodeManagerAddress"]]
                ):
                    self.resource.append(
                        {
                            "CPU": 0,
                            "GPU": resource_allocation["GPU"][
                                nodes["NodeManagerAddress"]
                            ],
                            "node": "node:" + nodes["NodeManagerAddress"],
                            "calculate_ability": 0,
                            "Used_count": 0.0,
                        }
                    )
                    print(self.resource[-1])
                    sum -= resource_allocation["GPU"][nodes["NodeManagerAddress"]]
                    # print("GPU", resource_allocation["GPU"][nodes['NodeManagerAddress']])

        for i in range(len(self.resource)):
            self.avaliable_resource.append(i)

        self.package_size = int((self.hyper_num - len(self.resource)) / 3)

    # Create a new trial
    def create_new_trial(self):
        self.trials_scheduler = sorted(self.trials_scheduler, reverse=False)
        self.trials_scheduler.sort(key=lambda t: self.trials_state[t]["iteration"])

        remaining_generations = 0
        for trial_state in self.trials_state:
            if trial_state["resource_id"] < 0:
                remaining_generations += math.ceil(
                    (self.max_iter - trial_state["iteration"])
                    / self.checkpoint_interval
                )

        cc = 1.0
        if len(self.trials_scheduler):
            while True:
                if not len(self.avaliable_resource):
                    break

                n = 1
                ids = []
                hypers = []
                checkpoints = []

                resource_id = self.avaliable_resource.pop(0)
                # Weak
                if (
                    self.resource[resource_id]["calculate_ability"] == 0
                    or self.resource[resource_id]["calculate_ability"]
                    > remaining_generations
                ):  # Assign poor resources to the fastest progressing trials
                    if (
                        self.stop_iteration
                        and self.resource[resource_id]["calculate_ability"]
                        > self.last_run_interval
                    ):
                        if (
                            (self.running_resource_num + 1) / self.package_size
                        ) < self.package_size:
                            self.package_size -= 1
                        continue
                    if self.stop_acc and self.max_acc >= self.stop_acc:
                        continue
                    ids.append(self.trials_scheduler.pop(-1))
                    if (
                        self.stop_iteration - self.trials_state[ids[0]]["iteration"]
                    ) < self.checkpoint_interval / 2:
                        self.checkpoints[ids[0]]["checkpoint_interval"] = (
                            self.stop_iteration - self.trials_state[ids[0]]["iteration"]
                        )
                    else:
                        self.checkpoints[ids[0]]["checkpoint_interval"] = int(
                            self.checkpoint_interval / 2
                        )
                        cc = 0.5
                    hypers.append(self.hypers[ids[0]])
                    checkpoints.append(self.checkpoints[ids[0]])
                # Strong
                else:
                    if (
                        len(self.trials_scheduler) >= self.package_size
                        and self.resource[resource_id]["calculate_ability"]
                        < self.package_size
                    ):
                        n = self.package_size

                    for i in range(n):
                        ids.append(self.trials_scheduler.pop(0))
                        if (
                            self.stop_iteration - self.trials_state[ids[i]]["iteration"]
                        ) < self.checkpoint_interval:
                            self.checkpoints[ids[i]]["checkpoint_interval"] = (
                                self.stop_iteration
                                - self.trials_state[ids[i]]["iteration"]
                            )
                        else:
                            self.checkpoints[ids[i]]["checkpoint_interval"] = (
                                self.checkpoint_interval
                            )
                        hypers.append(self.hypers[ids[i]])
                        checkpoints.append(self.checkpoints[ids[i]])

                # Training Remote
                Trial.options(
                    num_cpus=self.resource[resource_id]["CPU"],
                    num_gpus=self.resource[resource_id]["GPU"],
                    resources={self.resource[resource_id]["node"]: 0.1},
                ).remote(self.tuner, n, ids, hypers, checkpoints)

                self.running_trial_num += n
                self.running_resource_num += 1
                self.resource[resource_id]["Used_count"] += cc
                for i in range(n):
                    self.trials_state[ids[i]]["resource_id"] = resource_id
                break

    # Handle trials that have finished training
    def report_before_trial_end(self, n, ids, accuracys, run_times, checkpoints):
        for i in range(n):
            self.trial_acc_list[ids[i]] = accuracys[i]

            if (
                checkpoints[i]["checkpoint_interval"] >= self.checkpoint_interval
            ):  # Check if mutation interval is reached
                mutation.remote(
                    self.tuner,
                    ids[i],
                    self.hypers,
                    self.trial_acc_list,
                    self.last_checkpoint,
                    self.hyperparam_mutations,
                )

            resource_id = self.trials_state[ids[i]]["resource_id"]
            self.trials_state[ids[i]][
                "resource_id"
            ] = -1  # -1 indicates the trial is not running (resource released)

            self.trials_state[ids[i]]["run_time"] += run_times[i]
            self.trials_state[ids[i]]["iteration"] += checkpoints[i][
                "checkpoint_interval"
            ]
            self.checkpoints[i] = checkpoints[i]

            save_acc_to_json.remote(
                ids[i], accuracys[i], self.trials_state[ids[i]]["iteration"], self.path
            )

            if self.resource[resource_id]["Used_count"] == 0.5:
                self.min_run_one_interval_time = min(
                    self.min_run_one_interval_time, run_times[i]
                )
                calculate_ability = math.ceil(
                    run_times[i] / self.min_run_one_interval_time
                )
                for resource in self.resource:
                    if resource["calculate_ability"]:
                        self.resource[resource_id]["calculate_ability"] += int(
                            calculate_ability / resource["calculate_ability"]
                        )
                self.resource[resource_id]["calculate_ability"] += calculate_ability
                print(self.resource[resource_id])

            if self.trials_state[ids[i]]["iteration"] > self.max_iter:
                self.max_iter = self.trials_state[ids[i]]["iteration"]
                self.last_run_interval = int(
                    (self.stop_iteration - self.max_iter)
                    / self.checkpoint_interval
                    * self.hyper_num
                )

            if accuracys[i] > self.max_acc:
                self.max_acc = accuracys[i]

            check = 0
            if self.stop_iteration:
                if self.trials_state[ids[i]]["iteration"] < self.stop_iteration:
                    check += 1
                else:
                    check = -9  # Termination condition met

            if self.stop_acc != 1:
                if accuracys[i] < self.stop_acc:
                    check += 1
                else:
                    check = -9  # Termination condition met
            if check > 0:  # Add IDs that still need training to the scheduler
                self.trials_scheduler.append(ids[i])

            elif check < 0:
                self.trials_state[ids[i]][
                    "resource_id"
                ] = -2  # -2 indicates the trial has ended
            else:
                print("No end condition!!")  # No termination condition set
                exit(0)

        self.running_trial_num -= n
        self.running_resource_num -= 1
        self.avaliable_resource.append(resource_id)
        self.create_new_trial()  # Create new training processes

    # Check if all training is finished
    def is_finish(self):
        if len(self.trials_scheduler) + self.running_trial_num == 0:
            return True
        else:
            return False

    # Set the head reference
    def set_head(self, tuner):
        self.tuner = tuner

        for _ in range(len(self.resource)):
            self.create_new_trial()

    def set_after_mutation(self, id, chosed_id, new_hyper, last_checkpoint):
        self.last_checkpoint[id] = last_checkpoint

        if new_hyper:
            self.perturbs += 1
            self.hypers[id] = new_hyper
            self.checkpoints[id] = copy.deepcopy(self.checkpoints[chosed_id])

            # if self.trials_state[id]["iteration"] > self.trials_state[chosed_id]["iteration"]:
            #     if self.trials_state[id]["iteration"] == self.stop_iteration:
            #         self.trials_scheduler.append(id)
            #         self.trials_scheduler = sorted(self.trials_scheduler, reverse=False)
            #     self.trials_state[id]["iteration"] = self.trials_state[chosed_id]["iteration"]

    def get_for_reporter(self):
        return (
            self.hypers,
            self.trial_acc_list,
            self.trials_state,
            self.perturbs,
            self.running_trial_num,
            self.package_size,
        )

    def get_start_time(self):
        return self.start_time

    def get_resource(self):
        return self.resource

    def get_best_accuracy(self):
        max_list = list(
            map(self.trial_acc_list.index, heapq.nlargest(1, self.trial_acc_list))
        )
        return max_list[0], self.trial_acc_list[max_list[0]], self.perturbs


# Mutation
@ray.remote(num_cpus=0.1, resources={"node:" + HEAD_NODE_IP: 0.1})
def mutation(
    tuner,
    id,
    hypers,
    accuracys,
    last_checkpoint,
    hyperparam_mutations,
    resample_posibility=0.25,
    quantile_fraction=0.25,
):
    lower_quantile, upper_quantile = quantile(accuracys, quantile_fraction)
    if id in upper_quantile:
        last_checkpoint[id] = 1
    else:
        last_checkpoint[id] = 0

    new_hyper = None
    chosed_id = None

    if id in lower_quantile:  # Check if performance is poor
        print("--- Exploit ---")
        chosed_id = random.choice(upper_quantile)  # Select a successful trial
        print(
            f"Cloning  hyper_{str(chosed_id).zfill(5)} (score : {accuracys[chosed_id]}) to hyper_{str(id).zfill(5)} (score : {accuracys[id]}) \n"
        )
        if last_checkpoint[chosed_id] == 0:
            print(
                f"Hyper_{str(chosed_id).zfill(5)} doesn't have a checkpoint, skipping exploit for  hyper_{str(id).zfill(5)}!!"
            )
        else:
            new_hyper = explore(
                id, hypers[chosed_id], hyperparam_mutations, resample_posibility
            )  # Mutation

    tuner.set_after_mutation.remote(id, chosed_id, new_hyper, last_checkpoint[id])


# Find the upper and lower quantiles
def quantile(accuracys, quantile_fraction):
    trials = []
    for id, acc in enumerate(accuracys):
        if acc != 0:
            trials.append(id)

    if len(trials) <= 1:
        return [], []

    trials.sort(key=lambda t: accuracys[t])

    # Calculate number of trials in quantile
    num_trials_in_quantile = int(math.ceil(len(trials) * quantile_fraction))
    if num_trials_in_quantile > len(trials) / 2:
        num_trials_in_quantile = int(math.floor(len(trials) / 2))

    return (trials[:num_trials_in_quantile], trials[-num_trials_in_quantile:])


# Explore new hyperparameters
def explore(id, hyper, hyperparam_mutations, resample_posibility):
    new_hyper = hyper
    print(f"--- Explore the hyperparameters on  hyper_{str(id).zfill(5)} ---")
    for key, distribution in hyperparam_mutations.items():
        print(f"{key} : {hyper[key]} --- ", end="")
        if isinstance(distribution, list):
            if random.random() < resample_posibility or hyper[key] not in distribution:
                new_hyper[key] = random.choice(distribution)
                print(f"(resample)  --> {new_hyper[key]}")
            else:
                shift = random.choice([-1, 1])
                old_idx = distribution.index(hyper[key])
                new_idx = old_idx + shift
                new_idx = min(max(new_idx, 0), len(distribution) - 1)
                new_hyper[key] = distribution[new_idx]
                print(
                    f"(shift {'left' if shift == -1 else 'right'}) --> {new_hyper[key]}"
                )
        elif isinstance(distribution, tuple):
            if random.random() < resample_posibility:
                new_hyper[key] = random.uniform(distribution[0], distribution[1])
                print(f"(resample)  --> {new_hyper[key]}")
            else:
                mul = random.choice([0.8, 1.2])
                new_hyper[key] = hyper[key] * mul
                print(f"(* {mul})  --> {new_hyper[key]}")
    print()
    return new_hyper


@ray.remote(num_cpus=0.1, resources={"node:" + HEAD_NODE_IP: 0.1})
def save_acc_to_json(id, acc, iter, path):
    jsonFile = open(path + "/" + str(id) + "-accuracy.json", "a")
    data = {
        "iteration": iter,
        "accuracy": acc,
    }
    w = json.dumps(data)  # Generate data to write
    jsonFile.write(w)  # Write data
    jsonFile.write("\n")  # Write data
    jsonFile.close()


# Assigned a hyperparameter, handle training and data communication
@ray.remote
def Trial(tuner, n, ids, hypers, checkpoints):
    start_time = time.time()
    accs = []
    run_times = []

    model_type = hypers[0].get("model_type", "resnet-18")
    train_loader, test_loader = get_data_loader(
        model_type, hypers[0].get("batch_size", 512)
    )  # Create training data loaders

    if model_type == "resnet-18":
        # Create model
        model = models.resnet18()
        # Modify the output of the fully connected layer
        model.fc = nn.Linear(model.fc.in_features, 10)
    elif model_type == "resnet-50":
        # Create model
        model = models.resnet50()
        # Modify the output of the fully connected layer
        model.fc = nn.Linear(model.fc.in_features, 100)

    for i in range(n):
        if torch.cuda.is_available():
            device = torch.device("cuda")
            model = model.to(device)
            for k, v in checkpoints[i]["model_state_dict"].items():
                checkpoints[i]["model_state_dict"][k] = v.cuda()
            # Convert to GPU
            for state in checkpoints[i]["optimizer_state_dict"]["state"].values():
                for k, v in state.items():
                    state[k] = v.cuda()
        else:
            device = torch.device("cpu")

        model.load_state_dict(checkpoints[i]["model_state_dict"])
        optimizer = optim.SGD(
            model.parameters(),
            lr=hypers[i].get("lr", 0.01),
            momentum=hypers[i].get("momentum", 0.9),
        )
        optimizer.load_state_dict(checkpoints[i]["optimizer_state_dict"])
        checkpoints[i]["model_state_dict"] = None
        checkpoints[i]["optimizer_state_dict"] = None
        for param_group in optimizer.param_groups:
            if "lr" in hypers[i]:
                param_group["lr"] = hypers[i]["lr"]
            if "momentum" in hypers[i]:
                param_group["momentum"] = hypers[i]["momentum"]

        # Start training
        for _ in range(checkpoints[i]["checkpoint_interval"]):
            train(model, optimizer, train_loader, device)

        # Get accuracy
        accs.append(test(model, test_loader, device))

        # Start saving results
        run_times.append(time.time() - start_time)
        checkpoints[i]["model_state_dict"] = model.state_dict()
        checkpoints[i]["optimizer_state_dict"] = optimizer.state_dict()

        if torch.cuda.is_available():  # If using GPU, move model and optimizer to CPU because checkpoints are on CPU
            # Convert to CPU
            for k, v in checkpoints[i]["model_state_dict"].items():
                checkpoints[i]["model_state_dict"][k] = v.cpu()
            # Convert to CPU
            for state in checkpoints[i]["optimizer_state_dict"]["state"].values():
                for k, v in state.items():
                    state[k] = v.cpu()

    tuner.report_before_trial_end.remote(
        n, ids, accs, run_times, checkpoints
    )  # Send results back to Tuner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_times", type=int, default=1
    )  # Number of experiment repetitions
    args = parser.parse_args()

    dir_path = "/home/ray_cluster/Documents/workspace/tune_population_based/"

    # Initialize Ray
    runtime_env = {
        "working_dir": dir_path,
        "excludes": [
            "data/",
            "my_model/",
            "ray_results/",
            "pytorch-cifar/",
            "results/",
        ],
    }

    # Output used resources
    out_result = open(dir_path + "Running_Results.txt", "a+")
    out_result.write("+---------------+---------------+\n")
    out_result.write(f"{time.ctime()}  <<Our Results - {__file__}>> \n")
    out_result.write(f"Hyper_num = {HYPER_NUM} \n")
    out_result.write(f"Stop iteration = {STOP_ITER} \n")
    out_result.write(f"Stop accuracy = {STOP_ACC} \n")
    out_result.write(f"Checkpoint interval = {INTERVAL_CHECK} \n")
    out_result.write(f"Batch size = {BATCH_SIZE} \n")
    out_result.write(f"Resource allocation: {RESOURCE_ALLOCATION} \n")
    out_result.close()

    model_types = ["resnet-18", "resnet-50"]
    # model_types = ["resnet-50"]

    for model in model_types:
        out_result = open(dir_path + "Running_Results.txt", "a+")
        out_result.write(f"model_type: {model} \n")
        out_result.close()

        avg_run_time = 0
        avg_accuracy = 0

        for _ in range(args.exp_times):
            if ray.is_initialized():
                ray.shutdown()
            ray.init(
                address="ray://" + HEAD_NODE_IP + ":10001", runtime_env=runtime_env
            )
            # ray.init(address="auto", runtime_env=runtime_env)
            print(ray.available_resources())
            tt = time.ctime()
            tt_tmp = tt.split()
            json_path = (
                dir_path
                + "results/"
                + tt_tmp[-1]
                + "-"
                + tt_tmp[-4]
                + "-"
                + tt_tmp[-3]
                + "-"
                + tt_tmp[-2]
                + "/"
            )
            os.makedirs(json_path)
            print(f"{json_path = }")

            # Create Tuner
            tuner_head = Tuner.remote(
                hyper_num=HYPER_NUM,
                model_type=model,
                resource_allocation=RESOURCE_ALLOCATION,
                stop_acc=STOP_ACC,
                stop_iteration=STOP_ITER,
                checkpoint_interval=INTERVAL_CHECK,
                path=json_path,
                hyperparam_mutations={
                    "lr": (0.0001, 1),
                    "momentum": (0.0001, 1),
                },
            )

            tuner_head.set_head.remote(tuner_head)

            # Create Reporter
            Reporter.remote(
                tuner_head,
                max_report_frequency=INTERVAL_REPORT,
                hyper_num=HYPER_NUM,
            )

            # Wait here until all training is finished
            while not ray.get(tuner_head.is_finish.remote()):
                time.sleep(1)

            max_acc_index, max_acc, perturbs = ray.get(
                tuner_head.get_best_accuracy.remote()
            )

            start_time = ray.get(tuner_head.get_start_time.remote())
            avg_run_time += time.time() - start_time
            avg_accuracy += max_acc
            resource = ray.get(tuner_head.get_resource.remote())

            # Output results
            out_result = open(dir_path + "Running_Results.txt", "a+")
            out_result.write(f"Resource results: {resource} \n")
            out_result.close()
            ray.shutdown()
            time.sleep(10)

        # Output final results
        out_result = open(dir_path + "Running_Results.txt", "a+")
        out_result.write(f"Avg_total_runtime : {avg_run_time / args.exp_times} \n")
        out_result.write(f"Avg_accuracy : {avg_accuracy / args.exp_times} \n\n")
        out_result.close()
