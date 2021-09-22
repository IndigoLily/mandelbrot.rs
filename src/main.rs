extern crate jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;

// imports
mod colour;
mod interp;
mod progress;
mod settings;
mod utils;

use std::{
    env,
    io::Write,
    fs::{self, File},
    path::Path,
    sync::{Arc, Condvar, Mutex},
};

use num::Complex;
use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;

use std::f64::consts::TAU;

use colour::*;
use interp::*;
use progress::*;
use settings::SettingsBuilder;
use settings::Settings as Stg;
use utils::*;

type EscapeTime = Option<f64>;

const FRAMEDIR: &str = "frames";

// calculate the escape time at a point c in the complex plane
// if c is in the Mandelbrot set, returns None
fn calc_at(c: Complex<f64>, stg: &Stg) -> EscapeTime {
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

fn get_colour(escape: &EscapeTime, t: f64, stg: &Stg) -> Colour {
    if let Some(escape) = escape {
        match &stg.clr_algo {
            ColourAlgo::BW => colour::WHITE,

            ColourAlgo::Grey => {
                let val = (escape * 2.0 + t * TAU).sin() / 2.0 + 0.5;
                Colour::from([val; 3]).enc_gamma()
            }

            ColourAlgo::Bands(size) => {
                if (escape + t).rem_euclid(1.0) < *size {
		    WHITE
                } else {
		    BLACK
                }
            }

            ColourAlgo::Rgb => {
                let val = (escape + t) % 1.0;
                if val < 1.0 / 3.0 {
		    colour::RED
                } else if val < 2.0 / 3.0 {
		    colour::GREEN
                } else {
		    colour::BLUE
                }
            }

            ColourAlgo::SineMult(r, g, b) => [r,g,b].map(|c| (escape / 2.0 * c + t * TAU).sin() / 2.0 + 0.5).into(),

            ColourAlgo::SineAdd(r, g, b) => [r,g,b].map(|c| (escape * 2.0 + c * TAU + t * TAU).sin() / 2.0 + 0.5).into(),

            ColourAlgo::Palette(colours) => {
                let escape = escape + t * colours.len() as f64;
                let i1 = (escape as usize) % colours.len();
                let i2 = (i1 + 1) % colours.len();
                let i3 = (i1 + 2) % colours.len();
                let i0 = (i1 + colours.len() - 1) % colours.len();
                let percent = escape % 1.0;

                match &stg.interp {
                    Interpolation::None => colours[i1],

                    Interpolation::Linear => [
                        linear_interpolate(colours[i1].r, colours[i2].r, percent),
                        linear_interpolate(colours[i1].g, colours[i2].g, percent),
                        linear_interpolate(colours[i1].b, colours[i2].b, percent),
                    ].into(),

                    Interpolation::Cosine => [
                        cosine_interpolate(colours[i1].r, colours[i2].r, percent),
                        cosine_interpolate(colours[i1].g, colours[i2].g, percent),
                        cosine_interpolate(colours[i1].b, colours[i2].b, percent),
                    ].into(),

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
                    ].into(),

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
		    ].into(),
                }
            }
        }
    } else {
        stg.inside
    }
}

fn calc_aa(x: usize, y: usize, stg: &Stg) -> Vec<EscapeTime> {
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
fn avg_colours(escapes: &[EscapeTime], t: f64, stg: &Stg) -> Colour {
    escapes.iter().fold(Colour::default(), |clr, e| clr + get_colour(e, t, stg)) / stg.aa_sq_f64
}

fn render(stg: &Arc<Stg>) {
    let anim = stg.frames != 1;
    let pool = threadpool::Builder::new().build();

    // clean frame dir
    if anim {
        let framepath = Path::new("frames");
        if framepath.exists() {
            fs::remove_dir_all(framepath).unwrap();
        }
        fs::create_dir(framepath).unwrap();
    }

    // writer(s)
    let writers: Vec<_> = if anim {
        (0..stg.frames)
            .map(|frame| {
                let name = &format!("{}/{}.png", FRAMEDIR, frame);
                create_png_writer(name, stg)
            })
            .collect()
    } else {
        vec![create_png_writer("mandelbrot.png", stg)]
    };

    // (calc a row then render it to each frame) on each thread
    let writers: Arc<Vec<Mutex<_>>> =
        Arc::new(writers.into_par_iter().map(Mutex::new).collect());

    let progress = Arc::new(Progress::new("rendering", stg.height * stg.frames));

    #[allow(clippy::mutex_atomic)]
    let pixels_written = (0..stg.frames)
        .map(|_| (Condvar::new(), Mutex::new(0usize)))
        .collect::<Vec<_>>();
    let pixels_written = Arc::new(pixels_written);

    for y in 0..stg.height {
        let stg = Arc::clone(stg);
        let writers = Arc::clone(&writers);
        let progress = Arc::clone(&progress);
        let pixels_written = Arc::clone(&pixels_written);
        pool.execute(move || {
            let escapes = (0..stg.width)
                .map(|x| calc_aa(x, y, &stg))
                .collect::<Vec<Vec<EscapeTime>>>();

            for frame in 0..stg.frames {
                let t = frame as f64 / stg.frames_f64;
                let pix_row = escapes
                    .iter()
                    .map(|escapes| Pixel::from(avg_colours(escapes, t, &stg)))
                    .collect::<Vec<Pixel>>();

                // wait for (frame,i)
		#[allow(clippy::mutex_atomic)]
                let mut ready_y = pixels_written[frame].1.lock().unwrap();
                while *ready_y != y {
                    ready_y = pixels_written[frame].0.wait(ready_y).unwrap();
                }

                let mut w = writers[frame].lock().unwrap();
                for pix in pix_row {
                    w.write_all(&<[u8;3]>::from(pix)).unwrap();
                }
                drop(w); // explicitly drop to make lock available as soon as possible

                // say (frame,i+1) is ready
                *ready_y += 1;
                drop(ready_y);
                pixels_written[frame].0.notify_all();

                progress.inc();
            }
        });
    }
    pool.join();

    for writer in writers.iter() {
        writer.lock().unwrap().flush().unwrap();
    }

    Arc::try_unwrap(progress)
        .expect("All clones were given to pool, which has joined, so the count should be 1")
        .join();
}

fn main() {
    let mut stg = SettingsBuilder::default();

    // parse env vars
    {
        // if env var exists, parse and set rndr to that value
        macro_rules! setting {
            ($name:ident) => {
                if let Ok(var) = env::var(stringify!($name)) {
                    stg.$name = var.parse().unwrap();
                }
            };
        }

	setting!(width);
	setting!(height);
	setting!(frames);

	setting!(start_t);

	setting!(aa);
	setting!(max_itr);
	setting!(bail);

	setting!(zoom);
	setting!(degrees);
	setting!(ctr_x);
	setting!(ctr_y);

	setting!(julia);
	setting!(julia_ctr_x);
	setting!(julia_ctr_y);

	setting!(speed);
	setting!(acc);
	setting!(interp);
	setting!(inside);

        if let Ok(threads) = env::var("threads") {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads.parse().unwrap())
                .build_global()
                .unwrap();
        }

        if let Ok(clr_algo) = env::var("clr_algo") {
            let clr_algo = match clr_algo.to_lowercase().as_str() {
                "bw" => ColourAlgo::BW,
                "grey" => ColourAlgo::Grey,
                "rgb" => ColourAlgo::Rgb,
                "bands" => ColourAlgo::Bands(env_or_default("band_size", 0.5)),
		"sin_mult" => ColourAlgo::SineMult(
		    env_or_default("sin_r", 1.0),
		    env_or_default("sin_g", 1.0),
		    env_or_default("sin_b", 1.0),
		),
		"sin_add" => ColourAlgo::SineAdd(
		    env_or_default("sin_r", 1.0),
		    env_or_default("sin_g", 1.0),
		    env_or_default("sin_b", 1.0),
		),
                "palette" => ColourAlgo::Palette(Palette::from(env::var("palette").unwrap())),
                _ => panic!("Couldn't parse clr_algo setting"),
            };
            stg.clr_algo = clr_algo;
        }
    }

    serde_json::to_writer_pretty(File::create("settings.json").unwrap(), &stg).unwrap();

    let stg = stg.build();

    render(&Arc::new(stg));

    println!("Done");
}
