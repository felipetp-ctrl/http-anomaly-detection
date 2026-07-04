use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize)]
pub struct PredictRequest {
    pub ip: String,
    pub timestamp: f64,
    pub endpoint: String,
    pub method: String,
    pub status_code: i32,
    pub payload_size: f64,
    pub user_agent: String,
    pub response_time: f64,
}

#[derive(Serialize)]
pub struct PredictResponse {
    pub ip: String,
    pub is_anomaly: bool,
    pub anomaly_score: f64,
    pub top_features: HashMap<String, f64>,
    pub timing_ms: HashMap<String, f64>,
}
