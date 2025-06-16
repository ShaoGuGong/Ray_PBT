import json
from pathlib import Path

config = {}
path = Path.resolve(Path(Path(__file__).parent) / "../config.json")

if Path.exists(path):
    with Path(path).open("r") as f:
        config = json.load(f)

DATASET_PATH = config.get("dataset_path", "~/Documents/dataset/")
STOP_ACCURACY = config.get("stop_accuracy", 0.9)
STOP_ITERATION = config.get("stop_iteration", 1000)
PHASE_ITERATION = config.get("phase_iteration", 250)
MUTATION_ITERATION = config.get("mutation_iteration", 50)
GPU_MAX_ITERATION = config.get("gpu_max_iteration", 150)
TRIAL_RESULT_OUTPUT_PATH = Path(
    config.get("trial_result_output_path", "~/Documents/Ray_PBT/trial_result.output"),
).expanduser()
TRIAL_PROGRESS_OUTPUT_PATH = Path(
    config.get(
        "trial_progress_output_path",
        "~/Documents/Ray_PBT/trial_progress.output",
    ),
).expanduser()
