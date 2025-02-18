from pprint import pp

import ray


def fetch_nodes_resources():
    for node in ray.nodes():
        pp(node["NodeID"])
        pp(node["NodeName"])
        pp(node["Resources"])


class Worker:
    def __init__(self, node_address: str, num_cpu: int = 0, num_gpu: int = 0) -> None:
        self.num_cpu = num_cpu
        self.num_gpu = num_gpu
        self.node_address = node_address
        self.calculate_ability = 0
        self.used_count = 0.0

    def update_calculate_abiliity(self) -> None:
        pass


def generate_workers(config: dict) -> list[Worker]:

    return []
