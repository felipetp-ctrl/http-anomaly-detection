use std::collections::{HashMap, VecDeque};
use std::sync::Mutex;

const WINDOW_MAX: f64 = 300.0;

#[derive(Clone)]
pub struct RequestRecord {
    pub timestamp: f64,
    pub endpoint: String,
    pub status_code: i32,
    pub payload_size: f64,
    pub user_agent: String,
    pub response_time: f64,
}

pub struct IpState {
    records: Mutex<HashMap<String, VecDeque<RequestRecord>>>,
}

impl IpState {
    pub fn new() -> Self {
        Self {
            records: Mutex::new(HashMap::new()),
        }
    }

    pub fn add_record(&self, ip: &str, record: RequestRecord) {
        let mut map = self.records.lock().unwrap();
        let q = map.entry(ip.to_string()).or_insert_with(VecDeque::new);
        let now = record.timestamp;
        q.push_back(record);

        let cutoff = now - WINDOW_MAX;
        while let Some(front) = q.front() {
            if front.timestamp < cutoff {
                q.pop_front();
            } else {
                break;
            }
        }
        if q.is_empty() {
            map.remove(ip);
        }
    }

    pub fn get_records(&self, ip: &str, now: f64, window: f64) -> Vec<RequestRecord> {
        let map = self.records.lock().unwrap();
        match map.get(ip) {
            None => Vec::new(),
            Some(q) => {
                let cutoff = now - window;
                q.iter()
                    .filter(|r| r.timestamp >= cutoff)
                    .cloned()
                    .collect()
            }
        }
    }
}
