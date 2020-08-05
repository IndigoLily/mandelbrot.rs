use png::{BitDepth, ColorType, Encoder};
use std::fs::File;

use std::f64::consts::E;

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
    let bailout = 20.0;
    loop {
        z.sq().add(c);

        if z.mag_sq() > bailout {
            let itr = itr as f64;
            return Some(itr - (z.mag().log(E)/bailout.log(E)).log(2.0));
        }

        itr += 1;

        if itr >= max_itr {
            return None;
        }
    }
}

fn main() {
    let width = 1920;
    let height = 1080;
    let smaller = if width < height { width } else { height };

    let width_f = width as f64;
    let height_f = height as f64;
    let smaller_f = smaller as f64;

    let aa = 4;

    let mut data: Vec<Vec<f64>> = Vec::new();
    for y in 0..height {
        println!("{}/{}", y, height);
        for x in 0..width {

            let mut sub_pix: Vec<Vec<f64>> = Vec::new();
            for xaa in 0..aa {
                for yaa in 0..aa {
                    let aa = aa as f64;
                    let xaa = xaa as f64;
                    let yaa = yaa as f64;

                    let x = (x as f64 + xaa / aa - width_f  / 2.0) / smaller_f * 4.0;
                    let y = (y as f64 + yaa / aa - height_f / 2.0) / smaller_f * 4.0;

                    let c = Complex::new(x, y);
                    let escape = calc_at(&c);

                    sub_pix.push(match escape {
                        None => vec![0.0, 0.0, 0.0],
                        Some(escape) => vec![
                            (escape / 2.0 * 1.1).sin() / 2.0 + 0.5,
                            (escape / 2.0 * 1.2).sin() / 2.0 + 0.5,
                            (escape / 2.0 * 1.3).sin() / 2.0 + 0.5,
                        ],
                    });
                }
            }
            data.push(
                sub_pix
                    .iter()
                    .fold(vec![0.0, 0.0, 0.0], |acc, x| {
                        vec![
                          acc[0] + x[0],
                          acc[1] + x[1],
                          acc[2] + x[2],
                        ]
                    })
                    .iter()
                    .map(|x| { let aa = aa as f64; x / aa / aa })
                    .map(|x| x.powf(1.0/2.2))
                    .map(|x| x * 255.0)
                    .collect::<Vec<f64>>()
            );

        }
    }
    let data: Vec<u8> = data.iter().flatten().map(|&x| x as u8).collect();

    let file = File::create("mandelbrot.png").unwrap();

    let mut encoder = Encoder::new(file, width, height);
    encoder.set_color(ColorType::RGB);
    encoder.set_depth(BitDepth::Eight);

    let mut writer = encoder.write_header().unwrap();
    writer.write_image_data(&data).unwrap();
}
