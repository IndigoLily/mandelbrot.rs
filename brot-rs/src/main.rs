/*
extern crate jemallocator;

#[global_allocator]
static ALLOC: jemallocator::Jemalloc = jemallocator::Jemalloc;
*/

// imports
use std::{
    env,
    io::Write,
    fs::{self, File},
    path::Path,
    sync::{Arc, Condvar, Mutex},
};

use rayon::iter::IntoParallelIterator;
use rayon::prelude::*;

use librot_rs::*;
use librot_rs::colour::Pixel;
use settings::{ Settings as Stg, SettingsBuilder };

mod progress;
mod utils;

use progress::*;
use utils::*;

const FRAMEDIR: &str = "frames";

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
	let len = stg.frames.to_string().len();
        (0..stg.frames)
            .map(|frame| {
                let name = &format!("{}/{:0>len$}.png", FRAMEDIR, frame, len=len);
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
    let stgb = if let Some(path) = env::args().nth(1) {
	ron::de::from_reader(File::open(path).unwrap()).unwrap()
    } else {
	SettingsBuilder::default()
    };

    let stg = Arc::new(stgb.build());
    render(&stg);

    println!("Done");
}
