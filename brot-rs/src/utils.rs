use std::fmt::Debug;
use std::fs::File;
use std::str::FromStr;

use png::{BitDepth, ColorType, Encoder};

use crate::settings::Settings as Stg;

#[allow(dead_code)]
pub fn env_or_default<T>(name: &str, default: T) -> T
where
    T: FromStr,
    <T as FromStr>::Err: Debug,
{
    match std::env::var(name) {
        Ok(val) => val
            .parse()
            .unwrap_or_else(|_| panic!("Couldn't parse {} setting", name)),
        Err(_) => default,
    }
}

#[allow(dead_code)]
pub fn xy_iterator(width: usize, height: usize) -> Vec<(usize, usize)> {
    (0..height)
        .flat_map(move |y| (0..width).map(move |x| (x, y)))
        .collect()
}

pub fn create_png_writer<'a>(filename: &str, stg: &Stg) -> png::StreamWriter<'a, File> {
    let file = File::create(filename).expect("Couldn't open file");
    let mut encoder = Encoder::new(file, stg.width as u32, stg.height as u32);
    encoder.set_color(ColorType::Rgb);
    encoder.set_depth(BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .into_stream_writer()
        .unwrap()
}
