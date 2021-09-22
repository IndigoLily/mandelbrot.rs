use serde::{Serialize, Deserialize};
use std::ops::{ Add, AddAssign, Div, DivAssign, Index };
use std::str::FromStr;
use std::fs::File;
use std::io::{ BufReader, BufRead };

#[inline]
fn enc_gamma(n: f64) -> f64 {
    n.powf(1.0 / 2.2)
}

#[inline]
fn dec_gamma(n: f64) -> f64 {
    n.powf(2.2)
}

#[derive(Copy, Clone, Default, Serialize, Deserialize)]
#[serde(from="String", into="String")]
pub struct Colour { pub r: f64, pub g: f64, pub b: f64 }

impl Colour {
    pub fn enc_gamma(mut self) -> Self {
	self.r = enc_gamma(self.r);
	self.g = enc_gamma(self.g);
	self.b = enc_gamma(self.b);
	self
    }

    pub fn dec_gamma(mut self) -> Self {
	self.r = dec_gamma(self.r);
	self.g = dec_gamma(self.g);
	self.b = dec_gamma(self.b);
	self
    }
}

impl From<[f64;3]> for Colour {
    fn from(arr: [f64;3]) -> Self {
	Colour { r: arr[0], g: arr[1], b: arr[2] }
    }
}

impl From<String> for Colour {
    fn from(hex: String) -> Self {
	hex.parse().unwrap()
    }
}

impl From<Colour> for String {
    fn from(clr: Colour) -> Self {
	let clr = clr.enc_gamma();
	format!("#{:02X?}{:02X?}{:02X?}",
	    (clr.r * (u8::MAX as f64)) as u8,
	    (clr.g * (u8::MAX as f64)) as u8,
	    (clr.b * (u8::MAX as f64)) as u8,
	)
    }
}

#[derive(Debug)]
pub enum ParseColourError {
    UnsupportedLength,
    UnsupportedCharacter,
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
	    let hex: String = if len == 6 { hex.into() } else { hex.chars().flat_map(|c| [c;2]).collect() };
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

pub const BLACK: Colour = Colour { r: 0.0, g: 0.0, b: 0.0 };
pub const WHITE: Colour = Colour { r: 1.0, g: 1.0, b: 1.0 };



pub struct Pixel { r: u8, g: u8, b: u8 }

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

impl From<Pixel> for [u8;3] {
    #[inline]
    fn from(pix: Pixel) -> [u8;3] {
	[pix.r, pix.g, pix.b]
    }
}



#[derive(Serialize, Deserialize, Clone)]
#[serde(from="String",into="String")]
pub struct Palette {
    pub path: String,
    pub clrs: Vec<Colour>,
}

impl From<String> for Palette {
    fn from(path: String) -> Self {
	let file = File::open(&path).unwrap();
	let reader = BufReader::new(file);
	let clrs = reader.lines().map(|x| x.unwrap().parse().unwrap()).collect();
	Palette { path, clrs }
    }
}

impl From<Palette> for String {
    fn from(plt: Palette) -> Self {
	plt.path
    }
}

impl Index<usize> for Palette {
    type Output = Colour;
    fn index(&self, i: usize) -> &Self::Output {
	&self.clrs[i]
    }
}

impl Palette {
    pub fn len(&self) -> usize {
	self.clrs.len()
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub enum ColourAlgo {
    BW,
    Grey,
    Rgb,
    Bands(f64),
    SineMult(f64, f64, f64),
    SineAdd(f64, f64, f64),
    Palette(Palette),
}
