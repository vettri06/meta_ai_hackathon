import requests
from typing import Any, Dict, List, Optional

class FirewallClient:
    """Client for interacting with the Adaptive AI Firewall server."""
    
    def __init__(self, base_url: str = "http://localhost:7860"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/health").json()

    def reset(self, task: str = "easy", seed: Optional[int] = None) -> Dict[str, Any]:
        payload = {"task": task}
        if seed is not None:
            payload["seed"] = seed
        return requests.post(f"{self.base_url}/reset", json=payload).json()

    def step(self, actions: Dict[str, int]) -> Dict[str, Any]:
        return requests.post(f"{self.base_url}/step", json={"actions": actions}).json()

    def step_single(self, action: int) -> Dict[str, Any]:
        return requests.post(f"{self.base_url}/step_single", json={"action": action}).json()

    def state(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/state").json()

    def stats(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/stats").json()

    def list_tools(self) -> List[str]:
        return requests.get(f"{self.base_url}/tools").json().get("tools", [])

    def call_tool(self, name: str, kwargs: Dict[str, Any]) -> Any:
        return requests.post(f"{self.base_url}/tool/{name}", json={"kwargs": kwargs}).json()

    def schema(self) -> Dict[str, Any]:
        return requests.get(f"{self.base_url}/schema").json()
