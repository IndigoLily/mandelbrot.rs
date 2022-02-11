use serde::{Deserialize, Serialize};
use std::f64::consts::PI;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Interpolation {
    None,
    Linear,
    Cosine,
    Cubic,
    Hermite,
}

impl std::fmt::Display for Interpolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl std::str::FromStr for Interpolation {
    type Err = String;

    fn from_str(s: &str) -> Result<Interpolation, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" | "const" => Ok(Interpolation::None),
            "linear" | "lerp" => Ok(Interpolation::Linear),
            "cos" | "cosine" => Ok(Interpolation::Cosine),
            "cubic" => Ok(Interpolation::Cubic),
            "hermite" => Ok(Interpolation::Hermite),
            _ => Err("".into()),
        }
    }
}

pub fn linear_interpolate(a: f64, b: f64, c: f64) -> f64 {
    (b - a) * c + a
}

pub fn cosine_interpolate(y1: f64, y2: f64, mu: f64) -> f64 {
    let mu2 = (1.0 - (mu * PI).cos()) / 2.0;
    y1 * (1.0 - mu2) + y2 * mu2
}

pub fn cubic_interpolate(y0: f64, y1: f64, y2: f64, y3: f64, mu: f64) -> f64 {
    let mu2 = mu * mu;
    let a0 = y3 - y2 - y0 + y1;
    let a1 = y0 - y1 - a0;
    let a2 = y2 - y0;
    let a3 = y1;
    a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3
}

/// Tension: 1 is high, 0 normal, -1 is low
/// Bias: 0 is even, positive is towards first segment, negative towards the other
pub fn hermite_interpolate(
    y0: f64,
    y1: f64,
    y2: f64,
    y3: f64,
    mu: f64,
    tension: f64,
    bias: f64,
) -> f64 {
    let mu2 = mu * mu;
    let mu3 = mu2 * mu;

    let m0 = (y1 - y0) * (1.0 + bias) * (1.0 - tension) / 2.0;
    let m0 = m0 + (y2 - y1) * (1.0 - bias) * (1.0 - tension) / 2.0;

    let m1 = (y2 - y1) * (1.0 + bias) * (1.0 - tension) / 2.0;
    let m1 = m1 + (y3 - y2) * (1.0 - bias) * (1.0 - tension) / 2.0;

    let a0 = 2.0 * mu3 - 3.0 * mu2 + 1.0;
    let a1 = mu3 - 2.0 * mu2 + mu;
    let a2 = mu3 - mu2;
    let a3 = -2.0 * mu3 + 3.0 * mu2;

    a0 * y1 + a1 * m0 + a2 * m1 + a3 * y2
}
