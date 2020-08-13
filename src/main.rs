use png::{BitDepth, ColorType, Encoder};
use std::{
    env,
    fmt::Debug,
    fs::{self, File},
    io::Write,
    str::FromStr,
    sync::{mpsc, Arc},
};

use std::f64::consts::E;
use std::f64::consts::TAU;

type Pixel = Vec<u8>;

fn parse_var<T>(name: &str, default: T) -> T
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
        let width  = parse_var("width", 640);
        let height = parse_var("height", 640);
        let aa     = parse_var("aa", 1);
        let frames = parse_var("frames", 1);

        assert!(width  >= 1, "Width must be at least 1");
        assert!(height >= 1, "Height must be at least 1");
        assert!(aa     >= 1, "Anti-aliasing level must be at least 1");
        assert!(frames >= 1, "Frames must be at least 1");

        Arc::new(Settings::new(width, height, aa, frames))
    };

    if stg.frames > 1 {
        fs::remove_dir_all("frames").unwrap();
        fs::create_dir("frames").unwrap();
    }

    let pool = threadpool::ThreadPool::new(5);

    for frame in 0..stg.frames {
        let t = frame as f64 / stg.frames_f;

        let (tx, rx) = mpsc::channel();

        let mut data: Vec<Vec<Pixel>> = Vec::new();
        data.resize((stg.width * stg.height) as usize, Vec::new());

        for y in 0..stg.height {
            let tx = tx.clone();
            let stg = Arc::clone(&stg);

            pool.execute(move || {
                let mut row: Vec<Pixel> = Vec::with_capacity(stg.width as usize);
                for x in 0..stg.width {
                    row.push(get_pixel(x, y, t, &stg));
                }
                let val = (y, row);
                tx.send(val).unwrap();
            });
        }

        // need to drop original tx, or rx iterator will never end
        drop(tx);

        let mut count = 0;
        for msg in rx {
            let (y, pix) = msg;
            data[y as usize] = pix;
            count += 1;
            if stg.frames == 1 {
                print!("\r{}%", count * 100 / stg.height);
                std::io::stdout().flush().unwrap();
            }
        }

        let data: Vec<u8> = data
            .into_iter()
            .flatten()
            .collect::<Vec<Pixel>>()
            .into_iter()
            .flatten()
            .collect();

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
            .write_image_data(&data)
            .expect("Couldn't write to file");

        if stg.frames > 1 {
            print!("\r{}/{} frames", frame + 1, stg.frames);
            std::io::stdout().flush().unwrap();
        }
    }

    println!("\nDone");
}

#[allow(dead_code)]
struct Settings {
    width:     u32,
    height:    u32,
    smaller:   u32,
    aa:        u32,
    frames:    u32,

    width_f:   f64,
    height_f:  f64,
    smaller_f: f64,
    aa_f:      f64,
    frames_f:  f64,
}

impl Settings {
    fn new(width: u32, height: u32, aa: u32, frames: u32) -> Self {
        let smaller = if width < height { width } else { height };
        Settings {
            width,
            height,
            aa,
            smaller,
            frames,

            width_f:   width   as f64,
            height_f:  height  as f64,
            aa_f:      aa      as f64,
            smaller_f: smaller as f64,
            frames_f:  frames  as f64,
        }
    }
}

fn get_colour(escape: &Option<f64>, t: f64) -> Vec<f64> {
    if let Some(escape) = escape {
        /*
        let val = ((escape * 2.0 + t * TAU).sin() / 2.0 + 0.5).powf(2.2);
        vec![val, val, val]
        */

        /*
        let val = if (((escape * 10.0).log(10.0) * 10.0 - t) % 1.0) > 0.95 {
            1.0
        } else {
            0.0
        };
        vec![val, val, val]
        */

        /*
        vec![
            (escape / 2.0 * 1.1 + t * TAU).sin() / 2.0 + 0.5,
            (escape / 2.0 * 1.2 + t * TAU).sin() / 2.0 + 0.5,
            (escape / 2.0 * 1.3 + t * TAU).sin() / 2.0 + 0.5,
        ]
        */

        vec![
            (escape * 2.0 + TAU*1.0/2.0 + t * TAU).sin() / 2.0 + 0.5,
            (escape * 2.0 + TAU*1.0/3.0 + t * TAU).sin() / 2.0 + 0.5,
            (escape * 2.0 + TAU*1.0/4.0 + t * TAU).sin() / 2.0 + 0.5,
        ]
    } else {
        vec![0.0, 0.0, 0.0]
    }
}

fn get_pixel(x: u32, y: u32, t: f64, stg: &Settings) -> Vec<u8> {
    let mut sum: Vec<f64> = vec![0.0, 0.0, 0.0];
    let x = x as f64;
    let y = y as f64;
    for xaa in (0..stg.aa).map(|x| x as f64) {
        let nx = x + xaa / stg.aa_f;
        for yaa in (0..stg.aa).map(|x| x as f64) {
            let ny = y + yaa / stg.aa_f;

            let c = image_to_complex(nx, ny, stg);
            let esc = calc_at(&c);
            let colour = get_colour(&esc, t);

            sum[0] += colour[0];
            sum[1] += colour[1];
            sum[2] += colour[2];
        }
    }
    sum.iter()
        .map(|x| ((x / stg.aa_f / stg.aa_f).powf(1.0 / 2.2) * 255.0) as u8)
        .collect()
}

// take a point in image coordinates, and return its location in the complex plane
fn image_to_complex(x: f64, y: f64, stg: &Settings) -> Complex {
    let x = (x - stg.width_f  / 2.0) / stg.smaller_f * 4.0;
    let y = (y - stg.height_f / 2.0) / stg.smaller_f * 4.0;
    Complex::new(x, y)
}

struct Complex {
    real: f64,
    imag: f64,
}

impl Complex {
    fn new(real: f64, imag: f64) -> Self {
        Complex { real, imag }
    }

    fn add(&mut self, other: &Self) -> &mut Self {
        self.real += other.real;
        self.imag += other.imag;
        self
    }

    fn sq(&mut self) -> &mut Self {
        let tmp_real = self.real;
        self.real = self.real * self.real - self.imag * self.imag;
        self.imag = 2.0 * tmp_real * self.imag;
        self
    }

    fn mag_sq(&self) -> f64 {
        self.real * self.real + self.imag * self.imag
    }

    fn mag(&self) -> f64 {
        self.mag_sq().sqrt()
    }
}

fn calc_at(c: &Complex) -> Option<f64> {
    let mut z = Complex::new(0.0, 0.0);
    let mut itr = 0;
    let max_itr = 1000;
    let bailout = 400.0;
    loop {
        z.sq().add(c);

        if z.mag_sq() > bailout {
            let itr = itr as f64;
            return Some(itr - (z.mag().log(E) / bailout.log(E)).log(2.0));
        }

        itr += 1;

        if itr >= max_itr {
            return None;
        }
    }
}
