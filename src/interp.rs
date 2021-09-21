use serde::{Serialize,Deserialize};
use std::f64::consts::PI;

#[derive(Copy, Clone, Debug, Serialize, Deserialize)]
pub enum Interpolation {
    None,
    Linear,
    Cosine,
    Cubic,
}

impl std::str::FromStr for Interpolation {
    type Err = ();

    fn from_str(s: &str) -> Result<Interpolation, ()> {
        match s.to_lowercase().as_str() {
            "none" | "const" => Ok(Interpolation::None),
            "linear" | "lerp" => Ok(Interpolation::Linear),
            "cos" | "cosine" => Ok(Interpolation::Cosine),
            "cubic" => Ok(Interpolation::Cubic),
            _ => Err(()),
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
