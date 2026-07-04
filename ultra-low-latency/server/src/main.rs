mod features;
mod inference;
mod schemas;
mod state;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use actix_web::{web, App, HttpServer, HttpResponse};

use inference::ModelRunner;
use schemas::{PredictRequest, PredictResponse};
use state::{IpState, RequestRecord};

async fn root() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "service": "HTTP Anomaly Detection (Rust/ONNX)",
        "endpoints": ["/predict", "/health"]
    }))
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}

async fn predict(
    req: web::Json<PredictRequest>,
    ip_state: web::Data<IpState>,
    model: web::Data<ModelRunner>,
) -> HttpResponse {
    let t0 = Instant::now();

    let record = RequestRecord {
        timestamp: req.timestamp,
        endpoint: req.endpoint.clone(),
        status_code: req.status_code,
        payload_size: req.payload_size,
        user_agent: req.user_agent.clone(),
        response_time: req.response_time,
    };
    ip_state.add_record(&req.ip, record);
    let t_state = Instant::now();

    let records_30s = ip_state.get_records(&req.ip, req.timestamp, 30.0);
    let records_5min = ip_state.get_records(&req.ip, req.timestamp, 300.0);
    let feature_vec = features::compute_features(&records_30s, &records_5min);
    let t_features = Instant::now();

    let (is_anomaly, anomaly_score) = model.predict(&feature_vec);
    let t_predict = Instant::now();

    let feature_names = [
        "request_count_30s", "request_count_5min", "endpoint_entropy",
        "status_code_entropy", "status_401_ratio", "interval_std",
        "unique_ua_ratio", "known_ua_ratio", "payload_size_std", "response_time_std",
    ];
    let mut indexed: Vec<(usize, f64)> = feature_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    let top_features: HashMap<String, f64> = indexed.iter()
        .take(5)
        .map(|&(i, v)| (feature_names[i].to_string(), (v * 10000.0).round() / 10000.0))
        .collect();

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let mut timing = HashMap::new();
    timing.insert("state_update".to_string(), round_ms(t_state.duration_since(t0)));
    timing.insert("feature_calc".to_string(), round_ms(t_features.duration_since(t_state)));
    timing.insert("prediction".to_string(), round_ms(t_predict.duration_since(t_features)));
    timing.insert("total".to_string(), (total_ms * 100.0).round() / 100.0);

    let resp = PredictResponse {
        ip: req.ip.clone(),
        is_anomaly,
        anomaly_score: (anomaly_score * 10000.0).round() / 10000.0,
        top_features,
        timing_ms: timing,
    };

    HttpResponse::Ok().json(resp)
}

fn round_ms(d: std::time::Duration) -> f64 {
    (d.as_secs_f64() * 1000.0 * 100.0).round() / 100.0
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let assets_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
    let model = ModelRunner::new(&assets_dir).expect("Failed to load model artifacts");
    let model_data = web::Data::new(model);
    let ip_state = web::Data::new(IpState::new());

    println!("Ultra-low latency server starting on 0.0.0.0:8080");
    HttpServer::new(move || {
        App::new()
            .app_data(model_data.clone())
            .app_data(ip_state.clone())
            .route("/", web::get().to(root))
            .route("/health", web::get().to(health))
            .route("/predict", web::post().to(predict))
    })
    .bind("0.0.0.0:8080")?
    .run()
    .await
}
