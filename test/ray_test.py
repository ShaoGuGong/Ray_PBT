import ray
import requests

ray.init()

# 獲取 Ray Dashboard API 端點
dashboard_url = f"http://{ray.get_runtime_context().dashboard_url}/api/actors"

try:
    actors_info = requests.get(dashboard_url).json()
    print(actors_info)  # 這裡會顯示所有正在執行的 Actor
except Exception as e:
    print("Failed to fetch actor info:", e)
