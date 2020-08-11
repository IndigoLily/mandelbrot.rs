use png::{BitDepth, ColorType, Encoder};
use std::fs::File;
use std::io::Write;

use std::f64::consts::E;
use std::f64::consts::TAU;

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

fn main() {
    let width  = 1920/2;
    let height = 1080/2;
    let aa     = 2;
    let frames = 30;
    let stg    = Settings::new(width, height, aa, frames);

    for frame in 0..stg.frames {
        let t = frame as f64 / stg.frames_f;
        if frames > 1 {
            print!("Calculating frame {}/{}\r", frame, stg.frames);
            std::io::stdout().flush();
        }

        let mut data: Vec<u8> = Vec::with_capacity((stg.width * stg.height) as usize);
        for y in 0..stg.height {
            if frames == 1 {
                print!("Calculating {}%\r", y * 100 / stg.height);
                std::io::stdout().flush();
            }
            for x in 0..stg.width {
                data.append(&mut get_pixel(x, y, t, &stg));
            }
        }

        if frames == 1 {
            println!("Calculating 100%");
        }

        let file = File::create(
            if stg.frames == 1 {
                String::from("mandelbrot.png")
            } else {
                format!("frames/{}.png", frame)
            })
            .expect("Couldn't open file");

        let mut encoder = Encoder::new(file, stg.width, stg.height);
        encoder.set_color(ColorType::RGB);
        encoder.set_depth(BitDepth::Eight);

        let mut writer = encoder.write_header().unwrap();
        writer.write_image_data(&data)
            .expect("Couldn't write to file");
    }

    if frames > 1 {
        println!("Calculating frame {0}/{0}", stg.frames);
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
    sum.iter().map(|x| ((x / stg.aa_f / stg.aa_f).powf(1.0/2.2) * 255.0) as u8).collect()
}

fn get_colour(escape: &Option<f64>, t: f64) -> Vec<f64> {
    if let Some(escape) = escape {
        /*
        vec![
            (escape / 2.0 * 1.1 + t * TAU).sin() / 2.0 + 0.5,
            (escape / 2.0 * 1.2 + t * TAU).sin() / 2.0 + 0.5,
            (escape / 2.0 * 1.3 + t * TAU).sin() / 2.0 + 0.5,
        ]
        */
        vec![
            (escape / 2.0 + TAU*1.0/2.0 + t * TAU).sin() / 2.0 + 0.5,
            (escape / 2.0 + TAU*1.0/3.0 + t * TAU).sin() / 2.0 + 0.5,
            (escape / 2.0 + TAU*1.0/4.0 + t * TAU).sin() / 2.0 + 0.5,
        ]
    } else {
        vec![0.0, 0.0, 0.0]
    }
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
