#![allow(unused_imports)]

extern crate jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

use std::{
    env,
    fmt::Debug,
    fs::{ self, File },
    io::{ Write, BufRead, BufReader, BufWriter },
    sync::{ Arc, Mutex, RwLock },
    str::FromStr,
    path::Path,
    time::Duration,
};

use png::{BitDepth, ColorType, Encoder};
use num::Complex;
use rayon::prelude::*;

use std::f64::consts::E;
use std::f64::consts::TAU;
use std::f64::consts::PI;

type Pixel = [u8; 3];
type Colour = [f64; 3];
type EscapeTime = Option<f64>;

const BLACK: Colour = [0.0; 3];
const FRAMEDIR: &str = "frames";

// utility functions {{{
fn deg_to_rad(deg: f64) -> f64 {
    deg / 360.0 * TAU
}

fn rotate_complex(c: &Complex<f64>, angle: f64, origin: &Complex<f64>) -> Complex<f64> {
    let angle = deg_to_rad(angle);
    (c - origin) * Complex::new(angle.cos(), angle.sin()) + origin
}

#[allow(dead_code)]
fn xy_iterator(width: usize, height: usize) -> Vec<(usize,usize)> {
    (0..height).flat_map(move |y| (0..width).map(move |x| {(x,y)})).collect()
}
// }}}

#[allow(dead_code)]
struct Renderer { // {{{
    width:     usize,
    height:    usize,
    smaller:   usize,
    frames:    usize,

    aa:        usize,
    max_itr:   usize,

    width_f:   f64,
    height_f:  f64,
    smaller_f: f64,
    frames_f:  f64,
    start_t:   f64,

    aa_f:      f64,
    aa_sq:     f64,
    bail:      f64,
    bail_sq:   f64,

    zoom:      f64,
    angle:     f64,
    center:    Complex<f64>,

    julia:     bool,
    ctr_julia: Complex<f64>,

    clr_algo:  ColourAlgo,
    interp:    Interpolation,
    inside:    Colour,
    speed:     f64,
    acc:       f64,
} // }}}

impl Renderer { // {{{
    fn new() -> Self { // {{{
        Renderer {
            width       : 640,
            height      : 640,
            smaller     : 640,
            frames      : 1,

            aa          : 1,
            max_itr     : 100,

            width_f     : 640.0,
            height_f    : 640.0,
            smaller_f   : 640.0,
            frames_f    : 1.0,
            start_t     : 0.0,

            aa_f        : 1.0,
            aa_sq       : 1.0,
            bail        : 20.0,
            bail_sq     : 20.0 * 20.0,

            zoom        : 1.0,
            angle       : 0.0,
            center      : Complex::new(0.0, 0.0),

            julia       : false,
            ctr_julia   : Complex::new(0.0, 0.0),

            clr_algo    : ColourAlgo::SineAdd(1.1, 1.2, 1.3),
            interp      : Interpolation::Cosine,
            inside      : BLACK,
            speed       : 1.0,
            acc         : 1.0,
        }
    } // }}}

    // builder functions {{{
    fn width(&mut self, width: usize) -> &mut Self {
        self.width = width;
        self.width_f = self.width as f64;
        self.smaller = self.width.min(self.height);
        self.smaller_f = self.smaller as f64;
        self
    }

    fn height(&mut self, height: usize) -> &mut Self {
        self.height = height;
        self.height_f = self.height as f64;
        self.smaller = self.width.min(self.height);
        self.smaller_f = self.smaller as f64;
        self
    }

    fn frames(&mut self, frames: usize) -> &mut Self {
        self.frames = frames;
        self.frames_f = self.frames as f64;
        self
    }

    fn aa(&mut self, aa: usize) -> &mut Self {
        self.aa = aa;
        self.aa_f = self.aa as f64;
        self.aa_sq = (self.aa * self.aa) as f64;
        self
    }

    fn max_itr(&mut self, max_itr: usize) -> &mut Self {
        self.max_itr = max_itr;
        self
    }

    fn zoom(&mut self, zoom: f64) -> &mut Self {
        self.zoom = zoom;
        self
    }

    fn angle(&mut self, angle: f64) -> &mut Self {
        self.angle = angle;
        self
    }

    fn center(&mut self, center: Complex<f64>) -> &mut Self {
        self.center = center;
        self
    }

    fn julia(&mut self, julia: bool) -> &mut Self {
        self.julia = julia;
        self
    }

    fn ctr_julia(&mut self, ctr_julia: Complex<f64>) -> &mut Self {
        self.ctr_julia = ctr_julia;
        self
    }

    fn clr_algo(&mut self, clr_algo: ColourAlgo) -> &mut Self {
        self.clr_algo = clr_algo;
        self
    }

    fn interp(&mut self, interp: Interpolation) -> &mut Self {
        self.interp = interp;
        self
    }

    fn inside(&mut self, inside: Colour) -> &mut Self {
        self.inside = inside;
        self
    }

    fn speed(&mut self, speed: f64) -> &mut Self {
        self.speed = speed;
        self
    }

    fn acc(&mut self, acc: f64) -> &mut Self {
        self.acc = acc;
        self
    }

    fn bail(&mut self, bail: f64) -> &mut Self {
        self.bail = bail;
        self.bail_sq = self.bail * self.bail;
        self
    }

    fn start_t(&mut self, start_t: f64) -> &mut Self {
        self.start_t = start_t;
        self
    }
    // }}}

    // take a point in image coordinates, and return its location in the complex plane
    fn image_to_complex(&self, x: f64, y: f64) -> Complex<f64> { // {{{
        let c = (Complex::new(x, y) - Complex::new(self.width_f, self.height_f) / 2.0) / self.smaller_f * 4.0 / self.zoom;
        if self.julia {
            rotate_complex(&c, self.angle, &Complex::new(0.0, 0.0)) + self.ctr_julia
        } else {
            c + self.center
        }
    } // }}}

    // calculate the escape time at a point c in the complex plane
    // if c is in the Mandelbrot set, returns None
    fn calc_at(&self, c: &Complex<f64>) -> EscapeTime { // {{{
        let mut z = c.clone();
        let mut itr = 1;

        loop {
            z = z * z + if self.julia { &self.center } else { c };

            if z.norm_sqr() > self.bail_sq {
                let itr = itr as f64;
                return Some(itr - (z.norm().log(E) / self.bail.log(E)).log(2.0));
            }

            itr += 1;

            if itr >= self.max_itr {
                return None;
            }
        }
    } // }}}

    fn get_colour(&self, escape: &EscapeTime, t: f64) -> Colour { // {{{
        if let Some(escape) = escape {
            let escape = escape.powf(self.acc) * self.speed;

            match &self.clr_algo {
                ColourAlgo::BW => [1.0, 1.0, 1.0],

                ColourAlgo::Grey => {
                    let val = ((escape * 2.0 + t * TAU).sin() / 2.0 + 0.5).powf(2.2);
                    [val; 3]
                },

                ColourAlgo::Bands(size) => {
                    let val = if (escape + t).rem_euclid(1.0) < *size {
                        1.0
                    } else {
                        0.0
                    };
                    [val; 3]
                },

                ColourAlgo::RGB => {
                    let val = (escape + t) % 1.0;
                    if val < 1.0/3.0 {
                        [1.0, 0.0, 0.0]
                    } else if val < 2.0/3.0 {
                        [0.0, 1.0, 0.0]
                    } else {
                        [0.0, 0.0, 1.0]
                    }
                },

                ColourAlgo::SineMult(r, g, b) => {
                    [
                        (escape / 2.0 * r + t * TAU).sin() / 2.0 + 0.5,
                        (escape / 2.0 * g + t * TAU).sin() / 2.0 + 0.5,
                        (escape / 2.0 * b + t * TAU).sin() / 2.0 + 0.5,
                    ]
                },

                ColourAlgo::SineAdd(r, g, b) => {
                    [
                        (escape * 2.0 + TAU * r + t * TAU).sin() / 2.0 + 0.5,
                        (escape * 2.0 + TAU * g + t * TAU).sin() / 2.0 + 0.5,
                        (escape * 2.0 + TAU * b + t * TAU).sin() / 2.0 + 0.5,
                    ]
                },

                ColourAlgo::Palette(colours) => {
                    let escape = escape + t * colours.len() as f64;
                    let i1 = (escape as usize) % colours.len();
                    let i2 = (i1 + 1) % colours.len();
                    let i3 = (i1 + 2) % colours.len();
                    let i0 = (i1 + colours.len() - 1) % colours.len();
                    let percent = escape % 1.0;

                    match &self.interp {
                        Interpolation::None => colours[i1],

                        Interpolation::Linear => [
                            linear_interpolate(colours[i1][0], colours[i2][0], percent),
                            linear_interpolate(colours[i1][1], colours[i2][1], percent),
                            linear_interpolate(colours[i1][2], colours[i2][2], percent),
                        ],

                        Interpolation::Cosine => [
                            cosine_interpolate(colours[i1][0], colours[i2][0], percent),
                            cosine_interpolate(colours[i1][1], colours[i2][1], percent),
                            cosine_interpolate(colours[i1][2], colours[i2][2], percent),
                        ],

                        Interpolation::Cubic => [
                            cubic_interpolate(
                                colours[i0][0],
                                colours[i1][0],
                                colours[i2][0],
                                colours[i3][0],
                                percent,
                            ),
                            cubic_interpolate(
                                colours[i0][1],
                                colours[i1][1],
                                colours[i2][1],
                                colours[i3][1],
                                percent,
                            ),
                            cubic_interpolate(
                                colours[i0][2],
                                colours[i1][2],
                                colours[i2][2],
                                colours[i3][2],
                                percent,
                            ),
                            ]
                    }
                },
            }
        } else {
            self.inside
        }
    } // }}}

    fn calc_aa(&self, x: usize, y: usize) -> Vec<EscapeTime> { // {{{
        let (x, y) = (x as f64, y as f64);
        let mut samples = Vec::with_capacity(self.aa * self.aa);
        for xaa in 0..self.aa {
            for yaa in 0..self.aa {
                let xaa = xaa as f64 / self.aa_f;
                let yaa = yaa as f64 / self.aa_f;
                samples.push(self.calc_at(&self.image_to_complex(x + xaa, y + yaa)));
            }
        }
        samples
    } // }}}

    fn avg_colours(&self, escapes: &Vec<EscapeTime>, t: f64) -> Colour { // {{{
        let mut sum: Colour = Default::default();
        for clr in escapes.iter().map(|e| self.get_colour(e,t)) {
            for ch in 0..3 {
                sum[ch] += clr[ch];
            }
        }
        for ch in sum.iter_mut() {
            *ch /= self.aa_sq;
        }
        sum
    } // }}}

    fn colour_to_pixel(clr: Colour) -> Pixel { // {{{
        [
            (clr[0].powf(1.0 / 2.2) * 255.0) as u8,
            (clr[1].powf(1.0 / 2.2) * 255.0) as u8,
            (clr[2].powf(1.0 / 2.2) * 255.0) as u8,
        ]
    } // }}}

    fn create_png_writer(&self, filename: &str) -> BufWriter<png::StreamWriter<File>> { // {{{
        let file = File::create(filename).expect("Couldn't open file");
        let mut encoder = Encoder::new(file, self.width as u32, self.height as u32);
        encoder.set_color(ColorType::Rgb);
        encoder.set_depth(BitDepth::Eight);
        BufWriter::with_capacity(1024 * 1024 /*1MiB*/, encoder.write_header().unwrap().into_stream_writer().unwrap())
    } // }}}

    fn render(self) { // {{{
        Arc::new(self).render_arc();
    } // }}}

    fn render_arc(self: &Arc<Self>) { // {{{
        let anim = self.frames != 1;
        let frame_area = self.width * self.height;
        let _total_area = frame_area * self.frames;

        // clean frame dir {{{
        if anim {
            let framepath = Path::new("frames");
            if framepath.exists() {
                fs::remove_dir_all(framepath).unwrap();
            }
            fs::create_dir(framepath).unwrap();
        }
        // }}}

        // writer(s) {{{
        let writers: Vec<_> = if anim {
            (0..self.frames).map(|frame| {
                let name = &format!("{}/{}.png", FRAMEDIR, frame);
                self.create_png_writer(name)
            }).collect()
        } else {
            vec![self.create_png_writer("mandelbrot.png")]
        };
        // }}}

        // escapes {{{
        let esc_count = Arc::new(RwLock::new(0usize));

        let esc_count_jh = std::thread::spawn({
            let esc_count = Arc::clone(&esc_count);
            let total = self.height;
            move || {
                let t = Duration::from_millis(100);
                loop {
                    let count = *esc_count.read().unwrap();
                    print!("\r{}% calculated", count * 100 / total);
                    std::io::stdout().flush().unwrap();
                    if count >= total {
                        break;
                    } else {
                        std::thread::sleep(t);
                    }
                }
                println!();
            }
        });

        let escapes: Vec<Vec<EscapeTime>> = (0..self.height).into_par_iter().flat_map(|y| {
            let row: Vec<Vec<EscapeTime>> = (0..self.width).map(|x| {
                self.calc_aa(x,y)
            }).collect();
            *esc_count.write().unwrap() += 1;
            row
        }).collect();

        esc_count_jh.join().unwrap();
        assert_eq!(*esc_count.read().unwrap(), self.height);
        // }}}

        // colouring and writing {{{
        let clr_count = Arc::new(RwLock::new(0usize));

        let clr_count_jh = std::thread::spawn({
            let clr_count = Arc::clone(&clr_count);
            let total = self.height * self.frames;
            move || {
                let t = Duration::from_millis(100);
                loop {
                    let count = *clr_count.read().unwrap();
                    print!("\r{}% coloured", count * 100 / total);
                    std::io::stdout().flush().unwrap();
                    if count >= total {
                        break;
                    } else {
                        std::thread::sleep(t);
                    }
                }
                println!();
            }
        });

        writers.into_par_iter().enumerate().for_each(|(frame, mut writer)| {
            let t = frame as f64 / self.frames_f;
            for y in 0..self.height {
                let row: Vec<Pixel> = (0..self.width).into_iter().map(|x| {
                    let colour = self.avg_colours(&escapes[x + y * self.width],t);
                    Renderer::colour_to_pixel(colour)
                }).collect();

                for pixel in row.iter() {
                    writer.write_all(pixel).unwrap();
                }

                *clr_count.write().unwrap() += 1;
            }
        });

        clr_count_jh.join().unwrap();
        assert_eq!(*clr_count.read().unwrap(), self.height * self.frames);
        // }}}
    } // }}}
} // }}}

fn env_or_default<T>(name: &str, default: T) -> T // {{{
where
    T: std::str::FromStr,
    <T as std::str::FromStr>::Err: Debug,
{
    match env::var(name) {
        Ok(val) => val
            .parse()
            .expect(&format!("Couldn't parse {} setting", name)),
        Err(_) => default,
    }
} // }}}

// intertpolation {{{
enum Interpolation {
    None,
    Linear,
    Cosine,
    Cubic,
}

impl FromStr for Interpolation {
    type Err = ();

    fn from_str(s: &str) -> Result<Interpolation, ()> {
        match s.to_lowercase().as_str() {
            "none" | "const"  => Ok(Interpolation::None),
            "linear" | "lerp" => Ok(Interpolation::Linear),
            "cos" | "cosine"  => Ok(Interpolation::Cosine),
            "cubic"           => Ok(Interpolation::Cubic),
            _ => Err(()),
        }
    }
}

fn linear_interpolate(a: f64, b: f64, c: f64) -> f64 {
    (b-a)*c+a
}

fn cosine_interpolate(y1: f64, y2: f64, mu: f64) -> f64 {
    let mu2 = (1.0 - (mu * PI).cos()) / 2.0;
    y1 * (1.0 - mu2) + y2 * mu2
}

fn cubic_interpolate(y0: f64, y1: f64, y2: f64, y3: f64, mu: f64) -> f64 {
    let mu2 = mu * mu;
    let a0  = y3 - y2 - y0 + y1;
    let a1  = y0 - y1 - a0;
    let a2  = y2 - y0;
    let a3  = y1;
    a0 * mu * mu2 + a1 * mu2 + a2 * mu + a3
}
// }}}

// colour {{{
enum ColourAlgo {
    BW,
    Grey,
    Bands(f64),
    RGB,
    SineMult(f64, f64, f64),
    SineAdd(f64, f64, f64),
    Palette(Vec<Colour>),
}

fn hex_to_rgb(hex: String) -> Colour {
    let hex = hex.trim_start_matches('#').to_lowercase();
    if hex.len() != 6 {
        panic!("RGB hex wrong length");
    }
    [
        (u8::from_str_radix(&hex[0..2], 16).unwrap() as f64 / u8::MAX as f64).powf(2.2),
        (u8::from_str_radix(&hex[2..4], 16).unwrap() as f64 / u8::MAX as f64).powf(2.2),
        (u8::from_str_radix(&hex[4..6], 16).unwrap() as f64 / u8::MAX as f64).powf(2.2),
    ]
}

fn load_palette(path: String) -> Vec<Colour> {
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    reader.lines().map(|x| hex_to_rgb(x.unwrap())).collect()
}
// }}}

fn main() { // {{{
    let mut rndr = Renderer::new();

    { // {{{
        // if env var exists, parse and set rndr to that value
        macro_rules! setting {
            ($name:ident) => {
                if let Ok($name) = env::var(stringify!($name)) {
                    rndr.$name($name.parse().unwrap());
                }
            }
        }

        setting!(width);
        setting!(height);
        setting!(frames);
        setting!(start_t);
        setting!(aa);
        setting!(bail);
        setting!(max_itr);
        setting!(zoom);
        setting!(angle);
        setting!(julia);
        setting!(interp);
        setting!(speed);
        setting!(acc);

        if let Ok(threads) = env::var("threads") {
            rayon::ThreadPoolBuilder::new().num_threads(threads.parse().unwrap()).build_global().unwrap();
        }

        // doesn't work with setting! macro because of hex_to_rgb
        if let Ok(inside) = env::var("inside") {
            rndr.inside(hex_to_rgb(inside));
        }

        if let (Ok(x), Ok(y)) = (env::var("center_x"), env::var("center_y")) {
            let x = x.parse().unwrap();
            let y = y.parse().unwrap();
            rndr.center(Complex::new(x,y));
        }

        if let (Ok(x), Ok(y)) = (env::var("ctr_julia_x"), env::var("ctr_julia_y")) {
            let x = x.parse().unwrap();
            let y = y.parse().unwrap();
            rndr.ctr_julia(Complex::new(x,y));
        }

        let sin_r     = env_or_default("sin_r", 1.0);
        let sin_g     = env_or_default("sin_g", 1.0);
        let sin_b     = env_or_default("sin_b", 1.0);
        let band_size = env_or_default("band_size", 0.5);

        if let Ok(clr_algo) = env::var("clr_algo") {
            let clr_algo = match clr_algo.to_lowercase().as_str() {
                "bw"       => ColourAlgo::BW,
                "grey"     => ColourAlgo::Grey,
                "rgb"      => ColourAlgo::RGB,
                "bands"    => ColourAlgo::Bands(band_size),
                "sin_mult" => ColourAlgo::SineMult(sin_r, sin_g, sin_b),
                "sin_add"  => ColourAlgo::SineAdd(sin_r, sin_g, sin_b),
                "palette"  => ColourAlgo::Palette(load_palette(env::var("palette").unwrap())),
                _ => panic!("Couldn't parse clr_algo setting"),
            };
            rndr.clr_algo(clr_algo);
        }

        assert!(rndr.width  >= 1, "Width must be at least 1");
        assert!(rndr.height >= 1, "Height must be at least 1");
        assert!(rndr.frames >= 1, "Frames must be at least 1");
        assert!(rndr.aa     >= 1, "Anti-aliasing level must be at least 1");
        assert!(rndr.bail   >= 20.0, "Bailout must be at least 20");
        assert!((0.0..=1.0).contains(&band_size), "Band size must be between 0 and 1");
    } // }}}

    rndr.render();

    println!("Done");
} // }}}
