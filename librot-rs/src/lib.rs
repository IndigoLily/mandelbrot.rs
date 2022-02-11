use std::f64::consts::TAU;

use num::Complex;

pub mod colour;
pub mod interp;
pub mod settings;

use colour::*;
use interp::*;
use settings::Settings;

pub type EscapeTime = Option<f64>;
pub type Stg = Settings;

/// Takes a point in image coordinates, and returns its location in the complex plane
pub fn image_to_complex(x: f64, y: f64, stg: &Stg) -> Complex<f64> {
    let c = (Complex::new(x, y) - Complex::new(stg.width_f64, stg.height_f64) / 2.0) /
        stg.smaller_f64 *
        4.0 /
        stg.zoom;
    c * stg.rotation + if stg.julia { stg.julia_ctr } else { stg.center }
}

// calculate the escape time at a point c in the complex plane
// if c is in the Mandelbrot set, returns None
pub fn calc_at(c: Complex<f64>, stg: &Stg) -> EscapeTime {
    let mut z = c;
    let c = if stg.julia { stg.center } else { c };

    for itr in 1..=stg.max_itr {
        z = z * z + c;

        if z.norm_sqr() > stg.bail_sq {
            let smooth = itr as f64 - (z.norm().ln() / stg.bail_ln).log2();
            return Some(smooth.powf(stg.acc) * stg.speed);
        }
    }

    None
}

pub fn get_colour(escape: &EscapeTime, t: f64, stg: &Stg) -> Colour {
    if let Some(escape) = escape {
        match &stg.clr_algo {
            ColourAlgo::BW => colour::WHITE,

            ColourAlgo::Grey => {
                let val = (escape * 2.0 + t * TAU).sin() / 2.0 + 0.5;
                Colour::from([val; 3]).enc_gamma()
            },

            ColourAlgo::Bands(size) => {
                if (escape + t).rem_euclid(1.0) < *size {
                    WHITE
                } else {
                    BLACK
                }
            },

            ColourAlgo::Rgb => {
                let val = (escape + t).rem_euclid(1.0);
                if val < 1.0 / 3.0 {
                    colour::RED
                } else if val < 2.0 / 3.0 {
                    colour::GREEN
                } else {
                    colour::BLUE
                }
            },

            ColourAlgo::SineMult(r, g, b) => [r, g, b]
                .map(|c| (escape / 2.0 * c + t * TAU).sin() / 2.0 + 0.5)
                .into(),

            ColourAlgo::SineAdd(r, g, b) => [r, g, b]
                .map(|c| (escape * 2.0 + c * TAU + t * TAU).sin() / 2.0 + 0.5)
                .into(),

            ColourAlgo::Palette(colours) => {
                let escape = escape + t * colours.len() as f64;
                let i1 = (escape as usize).rem_euclid(colours.len());
                let i2 = (i1 + 1).rem_euclid(colours.len());
                let i3 = (i1 + 2).rem_euclid(colours.len());
                let i0 = (i1 + colours.len() - 1).rem_euclid(colours.len());
                let percent = escape.rem_euclid(1.0);

                match &stg.interp {
                    Interpolation::None => colours[i1],

                    Interpolation::Linear => [
                        linear_interpolate(colours[i1].r, colours[i2].r, percent),
                        linear_interpolate(colours[i1].g, colours[i2].g, percent),
                        linear_interpolate(colours[i1].b, colours[i2].b, percent),
                    ]
                    .into(),

                    Interpolation::Cosine => [
                        cosine_interpolate(colours[i1].r, colours[i2].r, percent),
                        cosine_interpolate(colours[i1].g, colours[i2].g, percent),
                        cosine_interpolate(colours[i1].b, colours[i2].b, percent),
                    ]
                    .into(),

                    Interpolation::Cubic => [
                        cubic_interpolate(
                            colours[i0].r,
                            colours[i1].r,
                            colours[i2].r,
                            colours[i3].r,
                            percent,
                        ),
                        cubic_interpolate(
                            colours[i0].g,
                            colours[i1].g,
                            colours[i2].g,
                            colours[i3].g,
                            percent,
                        ),
                        cubic_interpolate(
                            colours[i0].b,
                            colours[i1].b,
                            colours[i2].b,
                            colours[i3].b,
                            percent,
                        ),
                    ]
                    .into(),

                    Interpolation::Hermite => [
                        hermite_interpolate(
                            colours[i0].r,
                            colours[i1].r,
                            colours[i2].r,
                            colours[i3].r,
                            percent,
                            0.0,
                            0.0,
                        ),
                        hermite_interpolate(
                            colours[i0].g,
                            colours[i1].g,
                            colours[i2].g,
                            colours[i3].g,
                            percent,
                            0.0,
                            0.0,
                        ),
                        hermite_interpolate(
                            colours[i0].b,
                            colours[i1].b,
                            colours[i2].b,
                            colours[i3].b,
                            percent,
                            0.0,
                            0.0,
                        ),
                    ]
                    .into(),
                }
            },
        }
    } else {
        stg.inside
    }
}

pub fn calc_aa(x: usize, y: usize, stg: &Stg) -> Vec<EscapeTime> {
    let (x, y) = (x as f64, y as f64);
    let mut samples = Vec::with_capacity(stg.aa_sq);
    for xaa in 0..stg.aa {
        for yaa in 0..stg.aa {
            let xaa = xaa as f64 / stg.aa_f64;
            let yaa = yaa as f64 / stg.aa_f64;

            samples.push(calc_at(image_to_complex(x + xaa, y + yaa, stg), stg));
        }
    }
    samples
}

#[inline]
pub fn avg_colours(escapes: &[EscapeTime], t: f64, stg: &Stg) -> Colour {
    escapes
        .iter()
        .fold(Colour::default(), |clr, e| clr + get_colour(e, t, stg)) /
        stg.aa_sq_f64
}
