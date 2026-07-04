use std::collections::HashMap;
use crate::state::RequestRecord;

const KNOWN_UA_PATTERNS: &[&str] = &[
    "Mozilla/5.0", "Chrome/", "Safari/", "Firefox/",
    "Edg/", "OPR/", "Opera/",
];

fn is_known_user_agent(ua: &str) -> bool {
    KNOWN_UA_PATTERNS.iter().any(|p| ua.contains(p))
}

fn entropy(values: &[&str]) -> f64 {
    let n = values.len();
    if n == 0 {
        return 0.0;
    }
    let mut counts: HashMap<&str, usize> = HashMap::new();
    for v in values {
        *counts.entry(v).or_insert(0) += 1;
    }
    let nf = n as f64;
    -counts.values()
        .filter(|&&c| c > 0)
        .map(|&c| {
            let p = c as f64 / nf;
            p * p.log2()
        })
        .sum::<f64>()
}

fn std_dev(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let nf = n as f64;
    let mean = values.iter().sum::<f64>() / nf;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    variance.sqrt()
}

pub fn compute_features(records_30s: &[RequestRecord], records_5min: &[RequestRecord]) -> [f64; 10] {
    let rc_30 = records_30s.len() as f64;
    let rc_5 = records_5min.len();

    if rc_5 == 0 {
        return [0.0; 10];
    }

    let rc_5f = rc_5 as f64;

    let endpoints: Vec<&str> = records_5min.iter().map(|r| r.endpoint.as_str()).collect();
    let status_codes: Vec<String> = records_5min.iter().map(|r| r.status_code.to_string()).collect();
    let status_strs: Vec<&str> = status_codes.iter().map(|s| s.as_str()).collect();
    let user_agents: Vec<&str> = records_5min.iter().map(|r| r.user_agent.as_str()).collect();

    let endpoint_ent = entropy(&endpoints);
    let status_ent = entropy(&status_strs);

    let status_401_count = records_5min.iter().filter(|r| r.status_code == 401).count();
    let status_401_ratio = status_401_count as f64 / rc_5f;

    let mut timestamps: Vec<f64> = records_5min.iter().map(|r| r.timestamp).collect();
    timestamps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let intervals: Vec<f64> = timestamps.windows(2).map(|w| w[1] - w[0]).collect();
    let interval_std_val = std_dev(&intervals);

    let mut unique_uas: Vec<&str> = user_agents.clone();
    unique_uas.sort();
    unique_uas.dedup();
    let unique_ua_ratio = unique_uas.len() as f64 / rc_5f;

    let known_count = user_agents.iter().filter(|ua| is_known_user_agent(ua)).count();
    let known_ua_ratio = known_count as f64 / rc_5f;

    let payload_sizes: Vec<f64> = records_5min.iter().map(|r| r.payload_size).collect();
    let payload_std = std_dev(&payload_sizes);

    let response_times: Vec<f64> = records_5min.iter().map(|r| r.response_time).collect();
    let response_std = std_dev(&response_times);

    [
        rc_30,
        rc_5f,
        endpoint_ent,
        status_ent,
        status_401_ratio,
        interval_std_val,
        unique_ua_ratio,
        known_ua_ratio,
        payload_std,
        response_std,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::RequestRecord;

    fn make_record(ts: f64, endpoint: &str, status: i32, payload: f64, ua: &str, rt: f64) -> RequestRecord {
        RequestRecord {
            timestamp: ts,
            endpoint: endpoint.to_string(),
            status_code: status,
            payload_size: payload,
            user_agent: ua.to_string(),
            response_time: rt,
        }
    }

    #[test]
    fn test_empty_records() {
        let result = compute_features(&[], &[]);
        assert_eq!(result, [0.0; 10]);
    }

    #[test]
    fn test_single_record() {
        let r = make_record(1000.0, "/login", 401, 256.0, "python-requests/2.31", 45.0);
        let result = compute_features(&[r.clone()], &[r]);
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 1.0);
        assert_eq!(result[2], 0.0);
        assert_eq!(result[4], 1.0);
        assert_eq!(result[7], 0.0);
    }

    #[test]
    fn test_known_ua() {
        let r = make_record(1000.0, "/home", 200, 500.0,
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36", 50.0);
        let result = compute_features(&[r.clone()], &[r]);
        assert_eq!(result[7], 1.0);
    }

    #[test]
    fn test_multi_record_cross_validated() {
        // Reference values from Python lib/feature_engineering.py
        let records = vec![
            make_record(1000.0, "/login", 401, 256.0, "python-requests/2.31", 45.0),
            make_record(1000.5, "/login", 401, 256.0, "python-requests/2.31", 46.0),
            make_record(1001.0, "/api/users", 200, 512.0,
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36", 50.0),
        ];
        let result = compute_features(&records, &records);

        let expected = [
            3.0,
            3.0,
            0.918295834054490,
            0.918295834054490,
            0.666666666666667,
            0.0,
            0.666666666666667,
            0.333333333333333,
            147.801668912544187,
            2.645751311064591,
        ];

        for i in 0..10 {
            assert!(
                (result[i] - expected[i]).abs() < 1e-10,
                "feature[{}]: got {}, expected {}", i, result[i], expected[i]
            );
        }
    }
}
