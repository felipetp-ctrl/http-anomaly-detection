from pydantic import BaseModel


class PredictRequest(BaseModel):
    ip: str
    timestamp: float
    endpoint: str
    method: str
    status_code: int
    payload_size: float
    user_agent: str
    response_time: float


class PredictResponse(BaseModel):
    ip: str
    is_anomaly: bool
    anomaly_score: float
    top_features: dict[str, float]
    timing_ms: dict[str, float]
