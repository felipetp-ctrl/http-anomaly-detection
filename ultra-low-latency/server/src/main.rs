mod features;
mod inference;
mod schemas;
mod state;

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

use actix_web::{web, App, HttpServer, HttpResponse};

use inference::ModelRunner;
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
    <style>
      :root {
        color-scheme: dark;
      }
      html, body {
        margin: 0;
        background:
          radial-gradient(circle at top, rgba(30, 41, 59, 0.92), rgba(2, 6, 23, 1) 50%),
          #020617;
        color: #e2e8f0;
        font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      }
      #swagger-ui {
        background: transparent;
      }
      .swagger-ui .topbar {
        display: none;
      }
      .swagger-ui .info .title,
      .swagger-ui .info p,
      .swagger-ui .opblock-summary-description,
      .swagger-ui .opblock-description-wrapper p,
      .swagger-ui .parameter__name,
      .swagger-ui .parameter__type,
      .swagger-ui .response-col_status,
      .swagger-ui .response-col_description,
      .swagger-ui .btn,
      .swagger-ui label,
      .swagger-ui .tab li,
      .swagger-ui .model-title,
      .swagger-ui .model,
      .swagger-ui .renderedMarkdown {
        color: #e2e8f0;
      }
      .swagger-ui .info {
        margin: 28px 0 18px;
      }
      .swagger-ui .scheme-container,
      .swagger-ui .opblock,
      .swagger-ui .models,
      .swagger-ui .dialog-ux .modal-ux,
      .swagger-ui .execute-wrapper {
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.16);
        box-shadow: 0 24px 80px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(14px);
      }
      .swagger-ui .info .title {
        font-size: 40px;
        letter-spacing: -0.04em;
        font-weight: 800;
      }
      .swagger-ui .info .description {
        max-width: 860px;
      }
      .swagger-ui .scheme-container {
        margin: 0 0 22px;
      }
      .swagger-ui .btn.authorize,
      .swagger-ui .btn.try-out__btn,
      .swagger-ui .btn.execute,
      .swagger-ui .btn.cancel {
        border-radius: 999px;
      }
      .swagger-ui .opblock.opblock-get .opblock-summary-method,
      .swagger-ui .opblock.opblock-post .opblock-summary-method {
        border-radius: 999px;
      }
    </style>
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

fn run_prediction(ip: &str, record: &RequestRecord, model: &ModelRunner, ip_state: &IpState) -> (bool, f64, HashMap<String, f64>) {
    ip_state.add_record(ip, record.clone());
    let records_30s = ip_state.get_records(ip, record.timestamp, 30.0);
    let records_5min = ip_state.get_records(ip, record.timestamp, 300.0);
    let feature_vec = features::compute_features(&records_30s, &records_5min);
    let (is_anomaly, anomaly_score) = model.predict(&feature_vec);

    let feature_names = [
        "request_count_30s", "request_count_5min", "endpoint_entropy",
        "status_code_entropy", "status_401_ratio", "interval_std",
        "unique_ua_ratio", "known_ua_ratio", "payload_size_std", "response_time_std",
    ];
    let mut indexed: Vec<(usize, f64)> = feature_vec.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed.sort_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
    let top_features = indexed.iter()
        .take(5)
        .map(|&(i, v)| (feature_names[i].to_string(), (v * 10000.0).round() / 10000.0))
        .collect();

    (is_anomaly, (anomaly_score * 10000.0).round() / 10000.0, top_features)
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
        let (is_anomaly, anomaly_score, top_features) = run_prediction(&scenario_data.ip, record, &model, &ip_state);
        if is_anomaly {
            anomaly_count += 1;
            if first_anomaly_at.is_none() {
                first_anomaly_at = Some(index + 1);
            }
        }
        if index % sample_every == 0 || index + 1 == total {
            samples.push(DemoSample {
                request_number: index + 1,
                is_anomaly,
                anomaly_score,
                top_features,
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
