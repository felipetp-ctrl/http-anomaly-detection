mod features;
mod inference;
mod schemas;
mod state;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use actix_web::{web, App, HttpServer, HttpResponse};

use inference::{ModelRunner, PredictionDiagnostics};
use schemas::{DemoResponse, DemoSample, DemoScenarioSummary, PredictRequest, PredictResponse};
use state::{IpState, RequestRecord};

async fn root() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "service": "HTTP Anomaly Detection (Rust/ONNX)",
        "endpoints": ["/health", "/predict", "/demo", "/demo/{scenario}", "/docs", "/openapi.json"]
    }))
}

async fn health() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({"status": "ok"}))
}

async fn openapi_json() -> HttpResponse {
    HttpResponse::Ok().json(openapi_spec())
}

async fn docs() -> HttpResponse {
    HttpResponse::Ok()
        .content_type("text/html; charset=utf-8")
        .body(swagger_ui_html())
}

fn swagger_ui_html() -> String {
    r#"<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>HTTP Anomaly Detection API Docs</title>
    <link rel="stylesheet" href="https://unpkg.com/swagger-ui-dist@5/swagger-ui.css" />
  </head>
  <body>
    <div id="swagger-ui"></div>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
    <script src="https://unpkg.com/swagger-ui-dist@5/swagger-ui-standalone-preset.js"></script>
    <script>
      window.ui = SwaggerUIBundle({
        url: "/openapi.json",
        dom_id: '#swagger-ui',
        deepLinking: true,
        displayRequestDuration: true,
        docExpansion: 'list',
        filter: true,
        showExtensions: true,
        showCommonExtensions: true,
        presets: [SwaggerUIBundle.presets.apis, SwaggerUIStandalonePreset],
        layout: 'BaseLayout'
      });
    </script>
  </body>
</html>"#.to_string()
}

fn openapi_spec() -> serde_json::Value {
    serde_json::json!({
        "openapi": "3.0.3",
        "info": {
            "title": "HTTP Anomaly Detection API",
            "version": "1.0.0",
            "description": "Interactive docs for the Rust HTTP anomaly detection service."
        },
        "servers": [
            {"url": "/"}
        ],
        "paths": {
            "/": {
                "get": {
                    "summary": "Root metadata",
                    "responses": {
                        "200": {
                            "description": "Service metadata"
                        }
                    }
                }
            },
            "/health": {
                "get": {
                    "summary": "Health check",
                    "responses": {
                        "200": {
                            "description": "Service is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/HealthResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/predict": {
                "post": {
                    "summary": "Score a single request",
                    "requestBody": {
                        "required": true,
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/PredictRequest"}
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/PredictResponse"}
                                }
                            }
                        }
                    }
                }
            },
            "/demo": {
                "get": {
                    "summary": "List demo scenarios",
                    "responses": {
                        "200": {
                            "description": "Available scenarios",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DemoScenarioList"}
                                }
                            }
                        }
                    }
                }
            },
            "/demo/{scenario}": {
                "get": {
                    "summary": "Run a demo scenario",
                    "parameters": [
                        {
                            "name": "scenario",
                            "in": "path",
                            "required": true,
                            "schema": {
                                "type": "string",
                                "enum": ["legitimate", "credential_stuffing", "ddos", "malicious_bot"]
                            }
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Scenario result",
                            "content": {
                                "application/json": {
                                    "schema": {"$ref": "#/components/schemas/DemoResponse"}
                                }
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "HealthResponse": {
                    "type": "object",
                    "properties": {
                        "status": {"type": "string", "example": "ok"}
                    },
                    "required": ["status"]
                },
                "PredictRequest": {
                    "type": "object",
                    "required": ["ip", "timestamp", "endpoint", "method", "status_code", "payload_size", "user_agent", "response_time"],
                    "properties": {
                        "ip": {"type": "string", "example": "203.0.113.10"},
                        "timestamp": {"type": "number", "format": "float", "example": 1751500000.0},
                        "endpoint": {"type": "string", "example": "/api/products"},
                        "method": {"type": "string", "example": "GET"},
                        "status_code": {"type": "integer", "example": 200},
                        "payload_size": {"type": "number", "format": "float", "example": 5000.0},
                        "user_agent": {"type": "string", "example": "Mozilla/5.0 (...) Chrome/125.0.0.0 Safari/537.36"},
                        "response_time": {"type": "number", "format": "float", "example": 50.0}
                    }
                },
                "PredictResponse": {
                    "type": "object",
                    "properties": {
                        "ip": {"type": "string"},
                        "is_anomaly": {"type": "boolean"},
                        "anomaly_score": {"type": "number"},
                        "top_features": {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        },
                        "timing_ms": {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        }
                    }
                },
                "DemoScenarioSummary": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "title": {"type": "string"},
                        "description": {"type": "string"}
                    }
                },
                "DemoScenarioList": {
                    "type": "object",
                    "properties": {
                        "available_scenarios": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/DemoScenarioSummary"}
                        }
                    }
                },
                "DemoSample": {
                    "type": "object",
                    "properties": {
                        "request_number": {"type": "integer"},
                        "is_anomaly": {"type": "boolean"},
                        "anomaly_score": {"type": "number"},
                        "top_features": {
                            "type": "object",
                            "additionalProperties": {"type": "number"}
                        }
                    }
                },
                "DemoResponse": {
                    "type": "object",
                    "properties": {
                        "scenario": {"type": "string"},
                        "description": {"type": "string"},
                        "ip": {"type": "string"},
                        "total_requests": {"type": "integer"},
                        "anomalies_detected": {"type": "integer"},
                        "first_anomaly_at": {"type": ["integer", "null"]},
                        "samples": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/DemoSample"}
                        }
                    }
                }
            }
        }
    })
}

#[derive(Clone, Copy)]
enum ScenarioName {
    Legitimate,
    CredentialStuffing,
    Ddos,
    MaliciousBot,
}

impl ScenarioName {
    fn from_str(value: &str) -> Option<Self> {
        match value {
            "legitimate" => Some(Self::Legitimate),
            "credential_stuffing" => Some(Self::CredentialStuffing),
            "ddos" => Some(Self::Ddos),
            "malicious_bot" => Some(Self::MaliciousBot),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Legitimate => "legitimate",
            Self::CredentialStuffing => "credential_stuffing",
            Self::Ddos => "ddos",
            Self::MaliciousBot => "malicious_bot",
        }
    }

    fn title(self) -> &'static str {
        match self {
            Self::Legitimate => "Usuário legítimo",
            Self::CredentialStuffing => "Credential Stuffing",
            Self::Ddos => "L7 DDoS",
            Self::MaliciousBot => "Bot malicioso (scan)",
        }
    }

    fn description(self) -> &'static str {
        match self {
            Self::Legitimate => "10 GETs espaçados, status 200, browser real — deve permanecer NORMAL",
            Self::CredentialStuffing => "400 POSTs rápidos em /login, 90% status 401, bot UA — deve virar ANOMALY",
            Self::Ddos => "1200 requests em rajada, mix 200/503/504, browser UA — deve virar ANOMALY",
            Self::MaliciousBot => "200 GETs em endpoints sensíveis (/.env, /admin, …), UA rotativo — deve virar ANOMALY",
        }
    }
}

struct ScenarioData {
    ip: String,
    requests: Vec<RequestRecord>,
}

struct PredictionOutcome {
    is_anomaly: bool,
    anomaly_score: f64,
    top_features: HashMap<String, f64>,
    timing_ms: HashMap<String, f64>,
}

const CREDENTIAL_STUFFING_IP_MARKER: &str = "cred-198.51.100.77";
const CREDENTIAL_STUFFING_DEBUG_LIMIT: usize = 10;

fn scenario_meta() -> Vec<DemoScenarioSummary> {
    [
        ScenarioName::Legitimate,
        ScenarioName::CredentialStuffing,
        ScenarioName::Ddos,
        ScenarioName::MaliciousBot,
    ]
    .iter()
    .map(|scenario| DemoScenarioSummary {
        name: scenario.as_str().to_string(),
        title: scenario.title().to_string(),
        description: scenario.description().to_string(),
    })
    .collect()
}

fn pseudo_value(seed: usize) -> f64 {
    let value = seed.wrapping_mul(1_103_515_245).wrapping_add(12_345) % 10_000;
    value as f64 / 10_000.0
}

fn pseudo_range(seed: usize, min: f64, max: f64) -> f64 {
    min + (max - min) * pseudo_value(seed)
}

fn scenario_data(name: ScenarioName) -> ScenarioData {
    let base_ts = 1_751_500_000.0;

    match name {
        ScenarioName::Legitimate => {
            let ip = "demo-legit-203.0.113.10".to_string();
            let endpoints = ["/api/products", "/api/users", "/home", "/about", "/api/search", "/js/app.js", "/css/style.css", "/images/logo.png"];
            let ua = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36";
            let mut requests = Vec::new();
            for i in 0..10 {
                requests.push(RequestRecord {
                    timestamp: base_ts + (i as f64) * 35.0,
                    endpoint: endpoints[i % endpoints.len()].to_string(),
                    status_code: 200,
                    payload_size: 5000.0 + (i as f64) * 3000.0,
                    user_agent: ua.to_string(),
                    response_time: 50.0 + (i as f64) * 40.0,
                });
            }
            ScenarioData { ip, requests }
        }
        ScenarioName::CredentialStuffing => {
            let ip = "demo-cred-198.51.100.77".to_string();
            let ua = "python-requests/2.31";
            let mut requests = Vec::new();
            for i in 0..400 {
                requests.push(RequestRecord {
                    timestamp: base_ts + (i as f64) * 0.05,
                    endpoint: "/login".to_string(),
                    status_code: if i % 10 != 9 { 401 } else { 200 },
                    payload_size: 256.0 + (i % 3) as f64,
                    user_agent: ua.to_string(),
                    response_time: 45.0 + (i % 5) as f64 * 0.5,
                });
            }
            ScenarioData { ip, requests }
        }
        ScenarioName::Ddos => {
            let ip = "demo-ddos-192.0.2.200".to_string();
            let ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36";
            let endpoints = ["/api/checkout", "/api/search"];
            let status_cycle = [200, 200, 200, 503, 504];
            let mut requests = Vec::new();
            for i in 0..1200 {
                let status = status_cycle[i % status_cycle.len()];
                requests.push(RequestRecord {
                    timestamp: base_ts + (i as f64) * 0.015,
                    endpoint: endpoints[i % endpoints.len()].to_string(),
                    status_code: status,
                    payload_size: pseudo_range(i + 1, 200.0, 800.0),
                    user_agent: ua.to_string(),
                    response_time: if status == 200 {
                        pseudo_range(i + 11, 30.0, 70.0)
                    } else {
                        pseudo_range(i + 21, 600.0, 1000.0)
                    },
                });
            }
            ScenarioData { ip, requests }
        }
        ScenarioName::MaliciousBot => {
            let ip = "demo-bot-172.16.50.99".to_string();
            let endpoints = ["/.env", "/.git/config", "/admin", "/wp-login.php", "/phpmyadmin", "/api/users", "/.aws/credentials", "/server-status", "/actuator", "/graphql", "/debug", "/console", "/.svn/entries", "/backup.sql", "/wp-admin", "/config.php", "/.htaccess", "/xmlrpc.php", "/api/v1/debug", "/metrics", "/health", "/trace", "/dump"];
            let uas = [
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0",
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36 Edg/124.0.0.0",
                "Mozilla/5.0 (iPhone; CPU iPhone OS 17_5 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.5 Mobile/15E148 Safari/604.1",
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            ];
            let status_cycle = [200, 404, 404, 403, 200, 404];
            let mut requests = Vec::new();
            for i in 0..200 {
                requests.push(RequestRecord {
                    timestamp: base_ts + (i as f64) * 1.5,
                    endpoint: endpoints[i % endpoints.len()].to_string(),
                    status_code: status_cycle[i % status_cycle.len()],
                    payload_size: pseudo_range(i + 31, 10.0, 2500.0),
                    user_agent: uas[i % uas.len()].to_string(),
                    response_time: pseudo_range(i + 41, 1.0, 130.0),
                });
            }
            ScenarioData { ip, requests }
        }
    }
}

fn run_prediction(ip: &str, record: &RequestRecord, model: &ModelRunner, ip_state: &IpState) -> PredictionOutcome {
    let t0 = Instant::now();
    ip_state.add_record(ip, record.clone());
    let t_state = Instant::now();

    let records_30s = ip_state.get_records(ip, record.timestamp, 30.0);
    let records_5min = ip_state.get_records(ip, record.timestamp, 300.0);
    let feature_vec = features::compute_features(&records_30s, &records_5min);
    let t_features = Instant::now();

    let diagnostics = model.diagnose(&feature_vec);
    let (is_anomaly, anomaly_score) = (diagnostics.is_anomaly, diagnostics.score_samples);
    let t_predict = Instant::now();

    let mut indexed: Vec<(usize, f64)> = feature_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    let top_features = indexed.iter()
        .take(5)
        .map(|&(i, v)| (features::FEATURE_NAMES[i].to_string(), (v * 10000.0).round() / 10000.0))
        .collect();

    let request_number = records_5min.len();
    if ip.contains(CREDENTIAL_STUFFING_IP_MARKER) && request_number <= CREDENTIAL_STUFFING_DEBUG_LIMIT {
        let previous_feature_vec = if request_number > 1 {
            Some(features::compute_features(
                &records_30s[..records_30s.len() - 1],
                &records_5min[..records_5min.len() - 1],
            ))
        } else {
            None
        };
        print_credential_stuffing_debug(
            request_number,
            record,
            &records_30s,
            &records_5min,
            &feature_vec,
            previous_feature_vec.as_ref(),
            &diagnostics,
        );
    }

    let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let mut timing = HashMap::new();
    timing.insert("state_update".to_string(), round_ms(t_state.duration_since(t0)));
    timing.insert("feature_calc".to_string(), round_ms(t_features.duration_since(t_state)));
    timing.insert("prediction".to_string(), round_ms(t_predict.duration_since(t_features)));
    timing.insert("total".to_string(), (total_ms * 100.0).round() / 100.0);

    PredictionOutcome {
        is_anomaly,
        anomaly_score: (anomaly_score * 10000.0).round() / 10000.0,
        top_features,
        timing_ms: timing,
    }
}

fn print_credential_stuffing_debug(
    request_number: usize,
    record: &RequestRecord,
    records_30s: &[RequestRecord],
    records_5min: &[RequestRecord],
    feature_vec: &[f64; 10],
    previous_feature_vec: Option<&[f64; 10]>,
    diagnostics: &PredictionDiagnostics,
) {
    println!("[credential_stuffing debug] request #{}", request_number);
    println!("  request: timestamp={:.3} endpoint={} status={} payload_size={:.1} user_agent={} response_time={:.1}",
        record.timestamp, record.endpoint, record.status_code, record.payload_size, record.user_agent, record.response_time);
    println!("  feature_names: {}", features::FEATURE_NAMES.join(", "));
    println!("  raw_features: {}", format_feature_pairs(feature_vec));
    println!("  scaled_features: {}", format_feature_pairs(&diagnostics.scaled_features));
    println!("  score_samples={:.10} decision_function={:.10} predict()={} offset_={:.10}",
        diagnostics.score_samples,
        diagnostics.decision_function,
        if diagnostics.is_anomaly { -1 } else { 1 },
        diagnostics.offset);
    println!("  top_5_abs_z: {}", format_top_abs_z_scores(&diagnostics.scaled_features));
    match previous_feature_vec {
        Some(previous) => println!("  changed_since_previous: {}", format_feature_changes(previous, feature_vec)),
        None => println!("  changed_since_previous: n/a (first request)"),
    }
    println!("  window_stats: {}", format_window_stats(records_30s, records_5min));
    if request_number == 1 {
        println!("  first_request_driver_hypothesis: a fully cold window with status_401_ratio=1.0000, known_ua_ratio=0.0000, endpoint_entropy=0.0000, status_code_entropy=0.0000, unique_ua_ratio=1.0000, and zero interval variance is already far from the training baseline");
    }
}

fn format_feature_pairs(values: &[f64; 10]) -> String {
    features::FEATURE_NAMES
        .iter()
        .zip(values.iter())
        .map(|(name, value)| format!("{}={:.4}", name, value))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_top_abs_z_scores(scaled_features: &[f64; 10]) -> String {
    let mut indexed: Vec<(usize, f64)> = scaled_features.iter().enumerate().map(|(index, value)| (index, *value)).collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    indexed
        .into_iter()
        .take(5)
        .map(|(index, value)| format!("{}={:+.4} (|z|={:.4})", features::FEATURE_NAMES[index], value, value.abs()))
        .collect::<Vec<_>>()
        .join(", ")
}

fn format_feature_changes(previous: &[f64; 10], current: &[f64; 10]) -> String {
    let mut changes = Vec::new();
    for index in 0..current.len() {
        let delta = current[index] - previous[index];
        if delta.abs() > 1e-12 {
            changes.push(format!(
                "{}: {:.4} -> {:.4} ({:+.4})",
                features::FEATURE_NAMES[index],
                previous[index],
                current[index],
                delta,
            ));
        }
    }

    if changes.is_empty() {
        "no feature changes".to_string()
    } else {
        changes.join(", ")
    }
}

fn format_window_stats(records_30s: &[RequestRecord], records_5min: &[RequestRecord]) -> String {
    let request_count_30s = records_30s.len();
    let request_count_5min = records_5min.len();

    let mut endpoints = Vec::new();
    let mut status_counts: HashMap<i32, usize> = HashMap::new();
    let mut timestamps: Vec<f64> = Vec::new();
    let mut payload_sizes: Vec<f64> = Vec::new();
    let mut response_times: Vec<f64> = Vec::new();

    for record in records_5min {
        endpoints.push(record.endpoint.as_str());
        *status_counts.entry(record.status_code).or_insert(0) += 1;
        timestamps.push(record.timestamp);
        payload_sizes.push(record.payload_size);
        response_times.push(record.response_time);
    }

    timestamps.sort_by(|left, right| left.partial_cmp(right).unwrap());
    let intervals: Vec<f64> = timestamps.windows(2).map(|window| window[1] - window[0]).collect();
    let interval_mean = if intervals.is_empty() {
        0.0
    } else {
        intervals.iter().sum::<f64>() / intervals.len() as f64
    };
    let interval_std = std_dev(&intervals);
    let interval_min = intervals.iter().cloned().fold(f64::INFINITY, f64::min);
    let interval_max = intervals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    let endpoint_entropy = entropy(&endpoints);
    let unique_endpoints = {
        let mut values = endpoints.clone();
        values.sort();
        values.dedup();
        values.len()
    };

    let status_ratio = |status_code: i32| -> f64 {
        if request_count_5min == 0 {
            0.0
        } else {
            *status_counts.get(&status_code).unwrap_or(&0) as f64 / request_count_5min as f64
        }
    };

    let payload_std = std_dev(&payload_sizes);
    let response_std = std_dev(&response_times);

    format!(
        "request_count_30s={} request_count_5min={} unique_endpoints={} endpoint_entropy={:.4} status_ratios={{200:{:.4}, 401:{:.4}, 403:{:.4}, 404:{:.4}, 5xx:{:.4}}} interval_stats={{count:{}, mean:{:.4}, std:{:.4}, min:{:.4}, max:{:.4}}} payload_std={:.4} response_std={:.4}",
        request_count_30s,
        request_count_5min,
        unique_endpoints,
        endpoint_entropy,
        status_ratio(200),
        status_ratio(401),
        status_ratio(403),
        status_ratio(404),
        if request_count_5min == 0 {
            0.0
        } else {
            status_counts
                .iter()
                .filter(|(status_code, _)| **status_code >= 500)
                .map(|(_, count)| *count as f64)
                .sum::<f64>()
                / request_count_5min as f64
        },
        intervals.len(),
        interval_mean,
        interval_std,
        if intervals.is_empty() { 0.0 } else { interval_min },
        if intervals.is_empty() { 0.0 } else { interval_max },
        payload_std,
        response_std,
    )
}

fn entropy(values: &[&str]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for value in values {
        *counts.entry(*value).or_insert(0) += 1;
    }
    let n = n as f64;
    -counts.values().map(|count| {
        let probability = *count as f64 / n;
        probability * probability.log2()
    }).sum::<f64>()
}

fn std_dev(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let n = n as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|value| (value - mean).powi(2)).sum::<f64>() / (n - 1.0);
    variance.sqrt()
}

async fn demo_list() -> HttpResponse {
    HttpResponse::Ok().json(serde_json::json!({
        "available_scenarios": scenario_meta()
    }))
}

async fn demo_run(path: web::Path<String>, model: web::Data<ModelRunner>, ip_state: web::Data<IpState>) -> HttpResponse {
    let Some(scenario) = ScenarioName::from_str(path.as_str()) else {
        return HttpResponse::NotFound().json(serde_json::json!({
            "error": "unknown scenario",
            "available_scenarios": scenario_meta(),
        }));
    };

    let scenario_data = scenario_data(scenario);
    let total = scenario_data.requests.len();
    let sample_every = std::cmp::max(1, total / 20);
    let mut samples = Vec::new();
    let mut anomaly_count = 0usize;
    let mut first_anomaly_at: Option<usize> = None;

    for (index, record) in scenario_data.requests.iter().enumerate() {
        let outcome = run_prediction(&scenario_data.ip, record, &model, &ip_state);
        if outcome.is_anomaly {
            anomaly_count += 1;
            if first_anomaly_at.is_none() {
                first_anomaly_at = Some(index + 1);
            }
        }
        if index % sample_every == 0 || index + 1 == total {
            samples.push(DemoSample {
                request_number: index + 1,
                is_anomaly: outcome.is_anomaly,
                anomaly_score: outcome.anomaly_score,
                top_features: outcome.top_features,
            });
        }
    }

    HttpResponse::Ok().json(DemoResponse {
        scenario: scenario.title().to_string(),
        description: scenario.description().to_string(),
        ip: scenario_data.ip,
        total_requests: total,
        anomalies_detected: anomaly_count,
        first_anomaly_at,
        samples,
    })
}

async fn predict(
    req: web::Json<PredictRequest>,
    ip_state: web::Data<IpState>,
    model: web::Data<ModelRunner>,
) -> HttpResponse {
    let record = RequestRecord {
        timestamp: req.timestamp,
        endpoint: req.endpoint.clone(),
        status_code: req.status_code,
        payload_size: req.payload_size,
        user_agent: req.user_agent.clone(),
        response_time: req.response_time,
    };
    let outcome = run_prediction(&req.ip, &record, &model, &ip_state);

    let resp = PredictResponse {
        ip: req.ip.clone(),
        is_anomaly: outcome.is_anomaly,
        anomaly_score: outcome.anomaly_score,
        top_features: outcome.top_features,
        timing_ms: outcome.timing_ms,
    };

    HttpResponse::Ok().json(resp)
}

fn round_ms(d: std::time::Duration) -> f64 {
    (d.as_secs_f64() * 1000.0 * 100.0).round() / 100.0
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let assets_dir = std::env::var("ASSETS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets"));
    let model = ModelRunner::new(&assets_dir).expect("Failed to load model artifacts");
    let model_data = web::Data::new(model);
    let ip_state = web::Data::new(IpState::new());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let bind_addr = format!("0.0.0.0:{port}");

    println!("Ultra-low latency server starting on {bind_addr}");
    HttpServer::new(move || {
        App::new()
            .app_data(model_data.clone())
            .app_data(ip_state.clone())
            .route("/", web::get().to(root))
            .route("/health", web::get().to(health))
            .route("/openapi.json", web::get().to(openapi_json))
            .route("/docs", web::get().to(docs))
            .route("/demo", web::get().to(demo_list))
            .route("/demo/{scenario}", web::get().to(demo_run))
            .route("/predict", web::post().to(predict))
    })
    .bind(bind_addr)?
    .run()
    .await
}
