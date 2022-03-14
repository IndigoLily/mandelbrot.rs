use std::fmt::Display;
use std::marker::PhantomData;
use std::ops::{Add, AddAssign, Div, DivAssign, Index};
use std::str::FromStr;
use std::fs::File;
use std::io::{BufReader, BufRead};

use palette::rgb::Rgb;
use palette::rgb::channels::Argb;
use serde::{Serialize, Deserialize};

pub use palette::{Pixel, Srgb, rgb::RgbStandard, encoding};
pub type LinSrgb = palette::LinSrgb<f64>;

pub const BLACK: LinSrgb = palette::rgb::Rgb{ red: 0.0, green: 0.0, blue: 0.0, standard: PhantomData };
pub const WHITE: LinSrgb = palette::rgb::Rgb{ red: 1.0, green: 1.0, blue: 1.0, standard: PhantomData };
pub const RED:   LinSrgb = palette::rgb::Rgb{ red: 1.0, green: 0.0, blue: 0.0, standard: PhantomData };
pub const GREEN: LinSrgb = palette::rgb::Rgb{ red: 0.0, green: 1.0, blue: 0.0, standard: PhantomData };
pub const BLUE:  LinSrgb = palette::rgb::Rgb{ red: 0.0, green: 0.0, blue: 1.0, standard: PhantomData };

/*
#[derive(Debug)]
pub enum ParseColourError {
    UnsupportedLength,
    UnsupportedCharacter,
}

impl std::error::Error for ParseColourError {}

impl Display for ParseColourError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl From<std::num::ParseIntError> for ParseColourError {
    #[inline]
    fn from(_: std::num::ParseIntError) -> Self {
        ParseColourError::UnsupportedCharacter
    }
}

impl FromStr for Colour {
    type Err = ParseColourError;
    fn from_str(hex: &str) -> Result<Self, Self::Err> {
        let hex = hex.trim_start_matches('#');

        let len = hex.len();
        if let 6 | 3 = len {
            let hex: String = if len == 6 {
                hex.into()
            } else {
                hex.chars().flat_map(|c| [c; 2]).collect()
            };
            let mut int = u32::from_str_radix(&hex, 16)?;
            let mut clr_array = [0.0; 3];
            for rgb in clr_array.iter_mut().rev() {
                *rgb = (int % 0x100) as f64 / (u8::MAX as f64);
                int /= 0x100;
            }
            Ok(Colour::from(clr_array).dec_gamma())
        } else {
            Err(ParseColourError::UnsupportedLength)
        }
    }
}

impl Add for Colour {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        Colour {
            r: self.r + other.r,
            g: self.g + other.g,
            b: self.b + other.b,
        }
    }
}

impl AddAssign for Colour {
    fn add_assign(&mut self, other: Self) {
        self.r += other.r;
        self.g += other.g;
        self.b += other.b;
    }
}

impl Div<f64> for Colour {
    type Output = Self;
    fn div(self, n: f64) -> Self::Output {
        Colour {
            r: self.r / n,
            g: self.g / n,
            b: self.b / n,
        }
    }
}

impl DivAssign<f64> for Colour {
    fn div_assign(&mut self, n: f64) {
        self.r /= n;
        self.g /= n;
        self.b /= n;
    }
}


pub struct Pixel {
    r: u8,
    g: u8,
    b: u8,
}

impl From<Colour> for Pixel {
    fn from(clr: Colour) -> Self {
        let clr = clr.enc_gamma();
        Pixel {
            r: (clr.r * 255.0) as u8,
            g: (clr.g * 255.0) as u8,
            b: (clr.b * 255.0) as u8,
        }
    }
}

impl From<Pixel> for [u8; 3] {
    #[inline]
    fn from(pix: Pixel) -> [u8; 3] {
        [pix.r, pix.g, pix.b]
    }
}
*/

fn parse_clr(hex: &str) -> Result<LinSrgb, ()> {
    let hex = hex.trim_start_matches('#');

    let len = hex.len();

    if let 6 | 3 = len {
        let hex: String =
            if len == 6 {
                hex.into()
            } else {
                hex.chars().flat_map(|c| [c; 2]).collect()
            };

        let int =
            0xff000000 + 
            match u32::from_str_radix(&hex, 16) {
                Ok(int) => int,
                Err(_) => return Err(()),
            };

        use palette::rgb::Srgb;
        Ok(
            Srgb::from_format(Rgb::from_u32::<Argb>(int)).into_linear()
        )
    } else {
        Err(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub struct Palette {
    pub path: String,
    pub clrs: Vec<LinSrgb>,
}

/*
impl FromStr for Palette {
    type Err = !;

    fn from_str(path: &str) -> Result<Self, Self::Err> {
        let file = File::open(&path).unwrap();
        let reader = BufReader::new(file);
        let clrs = reader
            .lines()
            .map(|x| parse_clr(&x.unwrap()).unwrap())
            .collect();

        Ok(Palette { path: path.to_string(), clrs })
    }
}
*/

impl From<String> for Palette {
    fn from(path: String) -> Self {
        let file = File::open(&path).unwrap();
        let reader = BufReader::new(file);
        let clrs = reader
            .lines()
            .map(|x| parse_clr(&x.unwrap()).unwrap())
            .collect();
        Palette { path, clrs }
    }
}

impl From<Palette> for String {
    fn from(plt: Palette) -> Self {
        plt.path
    }
}

impl Index<usize> for Palette {
    type Output = LinSrgb;
    fn index(&self, i: usize) -> &Self::Output {
        &self.clrs[i]
    }
}

impl Palette {
    pub fn len(&self) -> usize {
        self.clrs.len()
    }
}



#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ColourAlgo {
    BW,
    Grey,
    Rgb,
    Bands(f64),
    SineMult(f64, f64, f64),
    SineAdd(f64, f64, f64),
    Palette(Palette),
}

impl FromStr for ColourAlgo {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        use regex::Regex;

        Ok(match s.to_lowercase().as_str() {
            "bw" => ColourAlgo::BW,
            "grey" | "gray" => ColourAlgo::Grey,
            "rgb" => ColourAlgo::Rgb,
            s => {
                return if let Some(caps) =
                    Regex::new(r"^bands\((\d+\.?\d*)\)$").unwrap().captures(s)
                {
                    if let Ok(f) = caps.get(1).unwrap().as_str().parse() {
                        Ok(ColourAlgo::Bands(f))
                    } else {
                        Err("Error parsing number in bands".into())
                    }
                } else if let Some(caps) =
                    Regex::new(r"^sinemult\((\d+\.?\d*),(\d+\.?\d*),(\d+\.?\d*)\)$")
                        .unwrap()
                        .captures(s)
                {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        caps.get(1).unwrap().as_str().parse(),
                        caps.get(2).unwrap().as_str().parse(),
                        caps.get(3).unwrap().as_str().parse(),
                    ) {
                        Ok(ColourAlgo::SineMult(r, g, b))
                    } else {
                        Err("Error parsing numbers in sinemult".into())
                    }
                } else if let Some(caps) =
                    Regex::new(r"^sineadd\((\d+\.?\d*),(\d+\.?\d*),(\d+\.?\d*)\)$")
                        .unwrap()
                        .captures(s)
                {
                    if let (Ok(r), Ok(g), Ok(b)) = (
                        caps.get(1).unwrap().as_str().parse(),
                        caps.get(2).unwrap().as_str().parse(),
                        caps.get(3).unwrap().as_str().parse(),
                    ) {
                        Ok(ColourAlgo::SineAdd(r, g, b))
                    } else {
                        Err("Error parsing numbers in sineadd".into())
                    }
                } else {
                    Err("".into())
                }
            },
        })
    }
}

impl Display for ColourAlgo {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self)
    }
}
