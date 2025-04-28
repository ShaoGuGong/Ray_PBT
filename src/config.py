import json
import os

config = {}
path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../config.json"))

if os.path.exists(path):
    with open(path, "r") as f:
        config = json.load(f)

DATASET_PATH = config.get("dataset_path", "~/Documents/dataset/")
STOP_ACCURACY = config.get("stop_accuracy", 0.9)
STOP_ITERATION = config.get("stop_iteration", 1000)
MUTATION_ITERATION = config.get("mutation_iteration", 50)
GPU_MAX_ITERATION = config.get("gpu_max_iteration", 150)
TRIAL_RESULT_OUTPUT_PATH = os.path.expanduser(
    config.get(
        "trial_result_output_path",
        "~/Documents/Ray_PBT/trial_result.output",
    )
)
TRIAL_PROGRESS_OUTPUT_PATH = os.path.expanduser(
    config.get(
        "trial_progress_output_path",
        "~/Documents/Ray_PBT/trial_progress.output",
    )
)
