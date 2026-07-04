use serde::Deserialize;
use std::path::Path;

#[derive(Deserialize)]
struct Tree {
    children_left: Vec<i64>,
    children_right: Vec<i64>,
    feature: Vec<i64>,
    threshold: Vec<f64>,
    n_node_samples: Vec<i64>,
}

#[derive(Deserialize)]
struct ModelParams {
    mean: Vec<f64>,
    scale: Vec<f64>,
    offset: f64,
    #[allow(dead_code)]
    max_samples: usize,
    #[allow(dead_code)]
    n_estimators: usize,
    average_path_length_max_samples: f64,
    trees: Vec<Tree>,
}

pub struct ModelRunner {
    scaler_mean: [f64; 10],
    scaler_scale: [f64; 10],
    offset: f64,
    avg_path_len: f64,
    trees: Vec<Tree>,
}

const EULER_MASCHERONI: f64 = 0.5772156649015329;

fn average_path_length(n: f64) -> f64 {
    if n <= 1.0 {
        return 0.0;
    }
    if n == 2.0 {
        return 1.0;
    }
    2.0 * ((n - 1.0).ln() + EULER_MASCHERONI) - 2.0 * (n - 1.0) / n
}

impl ModelRunner {
    pub fn new(assets_dir: &Path) -> Result<Self, Box<dyn std::error::Error>> {
        let json = std::fs::read_to_string(assets_dir.join("model_params.json"))?;
        let params: ModelParams = serde_json::from_str(&json)?;

        let mut scaler_mean = [0.0f64; 10];
        let mut scaler_scale = [0.0f64; 10];
        scaler_mean.copy_from_slice(&params.mean);
        scaler_scale.copy_from_slice(&params.scale);

        Ok(Self {
            scaler_mean,
            scaler_scale,
            offset: params.offset,
            avg_path_len: params.average_path_length_max_samples,
            trees: params.trees,
        })
    }

    pub fn predict(&self, features: &[f64; 10]) -> (bool, f64) {
        let mut scaled = [0.0f64; 10];
        for i in 0..10 {
            scaled[i] = (features[i] - self.scaler_mean[i]) / self.scaler_scale[i];
        }

        let mut total_path_length = 0.0;
        for tree in &self.trees {
            total_path_length += self.tree_path_length(tree, &scaled);
        }
        let avg = total_path_length / self.trees.len() as f64;

        // score_samples = -2^(-avg / c(max_samples))
        let score = -(2.0f64).powf(-avg / self.avg_path_len);
        let decision = score - self.offset;
        let is_anomaly = decision < 0.0;

        (is_anomaly, score)
    }

    fn tree_path_length(&self, tree: &Tree, x: &[f64; 10]) -> f64 {
        let mut node = 0usize;
        let mut depth = 0.0;

        loop {
            let left = tree.children_left[node];
            let right = tree.children_right[node];

            // Leaf node (children_left == -1)
            if left == -1 {
                let n = tree.n_node_samples[node] as f64;
                return depth + average_path_length(n);
            }

            let feature_idx = tree.feature[node] as usize;
            if x[feature_idx] <= tree.threshold[node] {
                node = left as usize;
            } else {
                node = right as usize;
            }
            depth += 1.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn test_model_loads_and_predicts() {
        let assets = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("assets");
        if !assets.join("model_params.json").exists() {
            eprintln!("Skipping: model_params.json not found. Run export_onnx.py first.");
            return;
        }
        let runner = ModelRunner::new(&assets).expect("failed to load model");

        let features = [1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let (is_anomaly, score) = runner.predict(&features);
        println!("is_anomaly={}, score={:.10}", is_anomaly, score);
        assert!(!is_anomaly, "expected normal for this input");
        assert!(
            (score - (-0.6262395132)).abs() < 0.001,
            "score should be near -0.626, got {}",
            score
        );
    }
}
