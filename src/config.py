import json
import os

path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config.json"))
if not os.path.exists(path):
    path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../config-example.json")
    )

with open(path, "r") as f:
    config = json.load(f)

DATASET_PATH = config["dataset_path"]
STOP_ACCURACY = config["stop_accuracy"]
MUTATION_ITERATION = config["mutation_iteration"]
