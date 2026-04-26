from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
# Standard OpenEnv types (if openenv-core is installed)
try:
    from openenv.core.env_server.types import Action, Observation
except ImportError:
    # Fallback if not installed
    class Action(BaseModel):
        pass
    class Observation(BaseModel):
        pass

# --- Custom Action/Observation classes as seen in video ---

class FirewallAction(Action):
    """Action for the AI Firewall environment."""
    action: int = Field(..., description="Action index: 0=ALLOW, 1=BLOCK, 2=INSPECT, 3=SANDBOX, 4=RATE_LIMIT, 5=QUARANTINE")
    session_id: Optional[str] = Field(None, description="Specific session to act upon")

class FirewallObservation(Observation):
    """Observation for the AI Firewall environment."""
    features: List[float] = Field(..., description="22-dimensional normalized feature vector")
    focus_session_id: Optional[str] = Field(None, description="ID of the session currently in focus")

# --- Original models from env/models.py ---

class ActionRecord(BaseModel):
    tick: int
    session_id: str
    action: int
    action_name: str
    malicious: bool
    reward: float
    components: Dict[str, float]

class ResetRequest(BaseModel):
    task: str = Field(default="easy", description="Task difficulty: easy, medium, hard")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")

class StepRequest(BaseModel):
    actions: Dict[str, int] = Field(default_factory=dict, description="Map of session_id to action index")

class StepSingleRequest(BaseModel):
    action: int = Field(..., description="Action index (0-5) for the current focus session")

class ToolRequest(BaseModel):
    kwargs: Dict[str, Any] = Field(default_factory=dict, description="Arguments for the tool call")

class StateResponse(BaseModel):
    episode_id: int
    task: str
    step_count: int
    current_tick: int
    observation_dim: int
    num_actions: int
    budget_remaining: float
    total_reward: float
    pending_session_count: int
    inspected_session_count: int
    pending_session_ids: List[str]
    inspected_session_ids: List[str]
    queue_length: int
    focus_session_id: Optional[str]
    focus_observation: List[float]

class StepResponse(BaseModel):
    reward: float
    done: bool
    state: StateResponse
    info: Dict[str, Any]

class StepSingleResponse(BaseModel):
    observation: List[float]
    reward: float
    done: bool
    state: StateResponse
    info: Dict[str, Any]

class EvaluateSessionResponse(BaseModel):
    session_id: str
    features: Dict[str, Any]
    observation: List[float]
    is_inspected: bool
    revealed_malicious: Optional[bool]
    expires_tick: int

class NetworkStatsResponse(BaseModel):
    episode_id: int
    task: str
    tick: int
    step_count: int
    total_reward: float
    budget_remaining: float
    budget_used_pct: float
    total_malicious: int
    total_benign: int
    detection_rate: float
    false_positive_rate: float
    efficiency: float
    early_detection_bonus: float
    cascade_prevention: float
    correct_allows: int
    inspections: int
    expired_malicious: int
    expired_benign: int
    # Extended metrics (from enhanced environment parameters)
    false_flag_accuracy: float = 0.0
    stealth_detection_rate: float = 0.0
    burst_ticks: int = 0
    false_flags_seen: int = 0
    stealth_attacks_seen: int = 0
    config_params: Dict[str, Any] = {}

class HealthResponse(BaseModel):
    status: str
    version: str

class ToolsListResponse(BaseModel):
    tools: List[str]

class TakeActionResponse(BaseModel):
    reward: float
    record: ActionRecord

class LLMChatRequest(BaseModel):
    prompt: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: Optional[str] = None

class LLMChatResponse(BaseModel):
    content: str
    model: str

class LLMConfigResponse(BaseModel):
    base_url: str
    model: str
    has_api_key: bool

class LLMTestResponse(BaseModel):
    ok: bool
    model: str
    content: str
