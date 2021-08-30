use std::{
    env,
    fmt::Debug,
    fs::{ self, File },
    io::{ Write, BufRead, BufReader },
    sync::{ mpsc, Arc },
    str::FromStr,
};

use png::{BitDepth, ColorType, Encoder};
use num::Complex;
use threadpool::ThreadPool;
use rand::{thread_rng, Rng};

use std::f64::consts::E;
use std::f64::consts::TAU;
use std::f64::consts::PI;

type Pixel = Vec<u8>;

type EscapeTime = Option<f64>;

mod matrix {
    pub struct Matrix<T> {
        width: usize,
        height: usize,
        area: usize,
        vec: Vec<T>,
    }

    impl<T> Matrix<T> {
        pub fn from_vec(vec: Vec<T>, width: usize) -> Self {
            let height = vec.len() / width;
            assert!(width == vec.len() / height);
            Matrix { width, height, area: width * height, vec }
        }

        fn index_at(&self, x: usize, y: usize) -> Option<usize> {
            if x < self.width && y < self.height {
                Some(y * self.width + x)
            } else {
                None
            }
        }

        pub fn get(&self, x: usize, y: usize) -> Option<&T> {
            if let Some(idx) = self.index_at(x, y) {
                unsafe {
                    Some(self.vec.get_unchecked(idx))
                }
            } else {
                None
            }
        }

        pub fn get_mut(&mut self, x: usize, y: usize) -> Option<&mut T> {
            if let Some(idx) = self.index_at(x, y) {
                unsafe {
                    Some(self.vec.get_unchecked_mut(idx))
                }
            } else {
                None
            }
        }
    }

    impl<T> From<Vec<Vec<T>>> for Matrix<T> {
        fn from(vecs: Vec<Vec<T>>) -> Self {
            let len = vecs[0].len();
            assert!(vecs.iter().all(|v| v.len() == len));
            Matrix::from_vec(vecs.into_iter().flatten().collect(), len)
        }
    }

    impl<T: Clone> Matrix<T> {
        pub fn with_default(width: usize, height: usize, default: T) -> Self {
            let area = width * height;
            Matrix { width, height, area, vec: vec![default; area] }
        }
    }

    impl<T: Clone + Default> Matrix<T> {
        pub fn new(width: usize, height: usize) -> Self {
            Matrix::with_default(width, height, Default::default())
        }
    }
}

use matrix::Matrix;

#[allow(dead_code)]
struct Settings {
    width:     u32,
    height:    u32,
    smaller:   u32,
    frames:    u32,

    aa:        u32,
    max_itr:   u32,

    width_f:   f64,
    height_f:  f64,
    smaller_f: f64,
    frames_f:  f64,
    start_t:   f64,

    aa_f:      f64,
    bail:      f64,
    bail_sq:   f64,

    zoom:      f64,
    angle:     f64,
    center:    Complex<f64>,

    julia:     bool,
    ctr_julia: Complex<f64>,

    clr_algo:  ColourAlgo,
    interp:    Interpolation,
    inside:    Vec<f64>,
    speed:     f64,
    acc:       f64,
}

impl Settings {
    fn new(
        width: u32, height: u32, frames: u32, start_t: f64,
        aa: u32, bail: f64, max_itr: u32,
        zoom: f64, angle: f64, center_x: f64, center_y: f64,
        julia: bool, ctr_julia_x: f64, ctr_julia_y: f64, 
        clr_algo: ColourAlgo, interp: Interpolation, inside: Vec<f64>, speed: f64, acc: f64,
        ) -> Self
    {
        let smaller = if width < height { width } else { height };
        Settings {
            width, height, smaller, frames, start_t,

            aa, max_itr,

            width_f:   width   as f64,
            height_f:  height  as f64,
            smaller_f: smaller as f64,
            frames_f:  frames  as f64,

            aa_f:    aa as f64,
            bail,
            bail_sq: bail * bail,

            zoom,
            angle,
            center: Complex::new(center_x, center_y),

            julia,
            ctr_julia: Complex::new(ctr_julia_x, ctr_julia_y),

            clr_algo, inside, interp, speed, acc,
        }
    }
}

fn env_or_default<T>(name: &str, default: T) -> T
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
}

fn main() {
    let stg = {
        let width       = env_or_default("width", 640);
        let height      = env_or_default("height", 640);
        let frames      = env_or_default("frames", 1);
        let start_t     = env_or_default("start", 0.0);

        let aa          = env_or_default("aa", 1);
        let bail        = env_or_default("bail", 20.0);
        let max_itr     = env_or_default("itr", 100);

        let zoom        = env_or_default("zoom", 1.0);
        let angle       = env_or_default("angle", 0.0);
        let center_x    = env_or_default("center_x", 0.0);
        let center_y    = env_or_default("center_y", 0.0);

        let julia       = env_or_default("julia", false);
        let ctr_julia_x = env_or_default("ctr_julia_x", 0.0);
        let ctr_julia_y = env_or_default("ctr_julia_y", 0.0);

        let inside      = hex_to_rgb(env_or_default("inside", "000000".to_string()));
        let interp      = env_or_default("interp", Interpolation::Cosine);
        let speed       = env_or_default("speed", 1.0);
        let acc         = env_or_default("acc", 1.0);

        let sin_r       = env_or_default("sin_r", 1.0);
        let sin_g       = env_or_default("sin_g", 1.0);
        let sin_b       = env_or_default("sin_b", 1.0);
        let band_size   = env_or_default("band_size", 0.5);

        let clr_algo = match env::var("clr_algo") {
            Ok(val) => match val.to_lowercase().as_str() {
                "bw"        => ColourAlgo::BW,
                "grey"      => ColourAlgo::Grey,
                "rgb"       => ColourAlgo::RGB,
                "bands"     => ColourAlgo::Bands(band_size),
                "sin_mult"  => ColourAlgo::SineMult(sin_r, sin_g, sin_b),
                "sin_add"   => ColourAlgo::SineAdd(sin_r, sin_g, sin_b),
                "palette"   => ColourAlgo::Palette(load_palette(env::var("palette").unwrap())),
                _ => panic!("Couldn't parse clr_algo setting"),
            },
            Err(_) => ColourAlgo::SineAdd(1.1, 1.2, 1.3),
        };

        assert!(width  >= 1, "Width must be at least 1");
        assert!(height >= 1, "Height must be at least 1");
        assert!(frames >= 1, "Frames must be at least 1");
        assert!(aa     >= 1, "Anti-aliasing level must be at least 1");
        assert!(bail   >= 20.0, "Bailout must be at least 20");
        assert!(0.0 <= band_size && band_size <= 1.0, "Band size must be between 0 and 1");

        Arc::new(
            Settings::new(
                width, height, frames, start_t,
                aa, bail, max_itr,
                zoom, angle, center_x, center_y,
                julia, ctr_julia_x, ctr_julia_y,
                clr_algo, interp, inside, speed, acc,
            )
        )
    };

    let pool = threadpool::ThreadPool::new(env_or_default("threads", 1));

    if stg.frames > 1 {
        fs::remove_dir_all("frames").unwrap();
        fs::create_dir("frames").unwrap();

        let escapes: Arc<Matrix<EscapeTime>> = Arc::new(calc_escapes(&stg, &pool));

        for frame in 0..stg.frames {
            // t is in the range [0, 1)
            let t = frame as f64 / stg.frames_f;

            let image_data: Vec<u8> = colourize(&escapes, t, &stg, &pool);

            let file = File::create(if stg.frames == 1 {
                String::from("mandelbrot.png")
            } else {
                format!("frames/{}.png", frame)
            })
            .expect("Couldn't open file");

            let mut encoder = Encoder::new(file, stg.width, stg.height);
            encoder.set_color(ColorType::RGB);
            encoder.set_depth(BitDepth::Eight);

            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&image_data)
                .expect("Couldn't write to file");

            print!("\r{}/{} frames", frame + 1, stg.frames);
            std::io::stdout().flush().unwrap();
        }
    } else {
        let (tx, rx) = mpsc::channel();

        for y in 0..stg.height {
            for x in 0..stg.width {
                let tx = tx.clone();
                let stg = stg.clone();
                pool.execute(move || {
                    let p = get_pixel(x, y, stg.start_t, &stg);
                    tx.send((x, y, p)).unwrap();
                });
            }
        }

        // need to drop original tx, or rx iterator will never end
        drop(tx);

        let mut image_data: Vec<u8> = vec![0; (stg.width * stg.height * 3) as usize];
        let mut count = 0;
        for msg in rx {
            let (x, y, p) = msg;
            let idx = ((stg.width * y + x) * 3) as usize;
            image_data[idx + 0] = p[0];
            image_data[idx + 1] = p[1];
            image_data[idx + 2] = p[2];
            count += 1;
            print!("\r{}% rendered", count * 100 / stg.width / stg.height);
            std::io::stdout().flush().unwrap();
        }

        let file = File::create("mandelbrot.png").expect("Couldn't open file");
        let mut encoder = Encoder::new(file, stg.width, stg.height);
        encoder.set_color(ColorType::RGB);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap();
        writer
            .write_image_data(&image_data)
            .expect("Couldn't write to file");
    }

    println!("\nDone");
}

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

enum ColourAlgo {
    BW,
    Grey,
    Bands(f64),
    RGB,
    SineMult(f64, f64, f64),
    SineAdd(f64, f64, f64),
    Palette(Vec<Vec<f64>>),
    //Iq([[f64; 3]; 4]),
}

fn hex_to_rgb(hex: String) -> Vec<f64> {
    let hex = hex.trim_start_matches('#').to_lowercase();
    if hex.len() != 6 {
        panic!("RGB hex wrong length");
    }
    vec![
        (u8::from_str_radix(&hex[0..2], 16).unwrap() as f64 / u8::MAX as f64).powf(2.2),
        (u8::from_str_radix(&hex[2..4], 16).unwrap() as f64 / u8::MAX as f64).powf(2.2),
        (u8::from_str_radix(&hex[4..6], 16).unwrap() as f64 / u8::MAX as f64).powf(2.2),
    ]
}

fn load_palette(path: String) -> Vec<Vec<f64>>{
    let file = File::open(path).unwrap();
    let reader = BufReader::new(file);
    reader.lines().map(|x| hex_to_rgb(x.unwrap())).collect()
}

fn get_colour(escape: &EscapeTime, t: f64, stg: &Settings) -> Vec<f64> {
    if let Some(escape) = escape {
        let escape = escape.powf(stg.acc) * stg.speed;

        match &stg.clr_algo {
            ColourAlgo::BW => vec![1.0, 1.0, 1.0],

            ColourAlgo::Grey => {
                let val = ((escape * 2.0 + t * TAU).sin() / 2.0 + 0.5).powf(2.2);
                vec![val, val, val]
            },

            ColourAlgo::Bands(size) => {
               let val = if (escape + t).rem_euclid(1.0) < *size {
                   1.0
               } else {
                   0.0
               };
               vec![val, val, val]
            },

            ColourAlgo::RGB => {
               let val = (escape + t) % 1.0;
               if val < 1.0/3.0 {
                   vec![1.0, 0.0, 0.0]
               } else if val < 2.0/3.0 {
                   vec![0.0, 1.0, 0.0]
               } else {
                   vec![0.0, 0.0, 1.0]
               }
            },

            ColourAlgo::SineMult(r, g, b) => {
                vec![
                    (escape / 2.0 * r + t * TAU).sin() / 2.0 + 0.5,
                    (escape / 2.0 * g + t * TAU).sin() / 2.0 + 0.5,
                    (escape / 2.0 * b + t * TAU).sin() / 2.0 + 0.5,
                ]
            },

            ColourAlgo::SineAdd(r, g, b) => {
                vec![
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

                match &stg.interp {
                    Interpolation::None => colours[i1].clone(),

                    Interpolation::Linear => vec![
                        linear_interpolate(colours[i1][0], colours[i2][0], percent),
                        linear_interpolate(colours[i1][1], colours[i2][1], percent),
                        linear_interpolate(colours[i1][2], colours[i2][2], percent),
                    ],

                    Interpolation::Cosine => vec![
                        cosine_interpolate(colours[i1][0], colours[i2][0], percent),
                        cosine_interpolate(colours[i1][1], colours[i2][1], percent),
                        cosine_interpolate(colours[i1][2], colours[i2][2], percent),
                    ],

                    Interpolation::Cubic => vec![
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
        stg.inside.clone()
    }
}

fn get_pixel(x: u32, y: u32, t: f64, stg: &Settings) -> Vec<u8> {
    let mut sum: Vec<f64> = vec![0.0, 0.0, 0.0];
    let mut rng = thread_rng();
    let x = x as f64;
    let y = y as f64;
    for _ in 0 .. stg.aa * stg.aa {
        let c = image_to_complex(x + rng.gen_range(0.0, 1.0), y + rng.gen_range(0.0, 1.0), &stg);
        let esc = calc_at(&c, stg);
        let colour = get_colour(&esc, t, stg);

        sum[0] += colour[0];
        sum[1] += colour[1];
        sum[2] += colour[2];
    }
    sum.iter()
        .map(|x| ((x / stg.aa_f / stg.aa_f).powf(1.0 / 2.2) * 255.0) as u8)
        .collect()
}

fn deg_to_rad(deg: f64) -> f64 {
    deg / 360.0 * TAU
}

fn rotate_complex(c: &Complex<f64>, angle: f64, origin: &Complex<f64>) -> Complex<f64> {
    let angle = deg_to_rad(angle);
    (c - origin) * Complex::new(angle.cos(), angle.sin()) + origin
}

// take a point in image coordinates, and return its location in the complex plane
fn image_to_complex(x: f64, y: f64, stg: &Settings) -> Complex<f64> {
    let c = (Complex::new(x, y) - Complex::new(stg.width_f, stg.height_f) / 2.0) / stg.smaller_f * 4.0 / stg.zoom;
    if stg.julia {
        rotate_complex(&c, stg.angle, &Complex::new(0.0, 0.0)) + stg.ctr_julia
    } else {
        c + stg.center
    }
}

// calculate the escape time at a point c in the complex plane
// if c is in the Mandelbrot set, returns None
fn calc_at(c: &Complex<f64>, stg: &Settings) -> EscapeTime {
    let mut z = c.clone();
    let mut itr = 1;

    loop {
        z = z * z + if stg.julia { &stg.center } else { c };

        if z.norm_sqr() > stg.bail_sq {
            let itr = itr as f64;
            return Some(itr - (z.norm().log(E) / stg.bail.log(E)).log(2.0));
        }

        itr += 1;

        if itr >= stg.max_itr {
            return None;
        }
    }
}

fn calc_escapes(stg: &Arc<Settings>, pool: &ThreadPool) -> Matrix<EscapeTime> {
    let (tx, rx) = mpsc::channel();

    let mut escapes: Vec<Vec<EscapeTime>> = Vec::new();
    escapes.resize((stg.height * stg.aa) as usize, Vec::new());

    for y in 0 .. stg.height * stg.aa {
        let tx = tx.clone();
        let stg = Arc::clone(stg);

        pool.execute(move || {
            let mut rng = thread_rng();
            let mut row: Vec<EscapeTime> = Vec::with_capacity((stg.width * stg.aa) as usize);
            for x in 0 .. stg.width * stg.aa {
                let c = image_to_complex((x / stg.aa) as f64 + rng.gen_range(0.0, 1.0), (y / stg.aa) as f64 + rng.gen_range(0.0, 1.0), &stg);
                row.push(calc_at(&c, &stg));
            }
            let val = (y, row);
            tx.send(val).unwrap();
        });
    }

    // need to drop original tx, or rx iterator will never end
    drop(tx);

    let mut count = 0;
    for msg in rx {
        let (y, row) = msg;
        escapes[y as usize] = row;
        count += 1;

        print!("\r{}% calculated", count * 100 / stg.height / stg.aa);
        std::io::stdout().flush().unwrap();
    }

    println!();

    Matrix::from(escapes)
}

fn colourize(escapes: &Arc<Matrix<EscapeTime>>, t: f64, stg: &Arc<Settings>, pool: &ThreadPool) -> Vec<u8> {
    let (tx, rx) = mpsc::channel();

    let mut data: Vec<Vec<Pixel>> = Vec::new();
    data.resize(stg.height as usize, Vec::new());

    for y in 0..stg.height {
        let tx = tx.clone();
        let stg = Arc::clone(&stg);
        let escapes = Arc::clone(&escapes);

        pool.execute(move || {
            let mut pix_row: Vec<Pixel> = Vec::with_capacity(stg.width as usize);
            for x in 0..stg.width {
                let mut sum: Vec<f64> = vec![0.0, 0.0, 0.0];

                for yaa in 0..stg.aa {
                    for xaa in 0..stg.aa {
                        let colour = get_colour(escapes.get((y * stg.aa + yaa) as usize, (x * stg.aa + xaa) as usize).unwrap(), t, &stg);
                        sum[0] += colour[0];
                        sum[1] += colour[1];
                        sum[2] += colour[2];
                    }
                }

                pix_row.push(sum.iter().map(|x| ((x / stg.aa_f / stg.aa_f).powf(1.0 / 2.2) * 255.0) as u8).collect::<Pixel>());
            }
            let val = (y, pix_row);
            tx.send(val).unwrap();
        });
    }

    // need to drop original tx, or rx iterator will never end
    drop(tx);

    let mut count = 0;
    for msg in rx {
        let (y, row) = msg;
        data[y as usize] = row;
        count += 1;

        if stg.frames == 1 {
            print!("\r{}% colourized", count * 100 / stg.height);
            std::io::stdout().flush().unwrap();
        }
    }

    if stg.frames == 1 {
        println!();
    }

    data.into_iter()
        .flatten()
        .collect::<Vec<Pixel>>()
        .into_iter()
        .flatten()
        .collect()
}
