use num::Complex;
use palette::Pixel;
//use serde::{Serialize, Deserialize};

use crate::colour::{ColourAlgo, LinSrgb, Palette};
use crate::interp::Interpolation;

pub use clap::Parser;

//#[derive(Serialize, Deserialize, Parser)]
//#[serde(default)]
#[derive(Parser)]
#[clap(global_setting = clap::AppSettings::DeriveDisplayOrder)]
pub struct SettingsBuilder {
    #[clap(short, long, default_value_t = 1920)]
    pub width: usize,

    #[clap(short, long, default_value_t = 1080)]
    pub height: usize,

    #[clap(short = 'x', long, default_value_t)]
    pub ctr_x: f64,

    #[clap(short = 'y', long, default_value_t)]
    pub ctr_y: f64,

    #[clap(short, long, default_value_t = 1.0)]
    pub zoom: f64,

    #[clap(short, long, default_value_t)]
    pub degrees: f64,

    #[clap(short = 'a', long, default_value_t = 1, name = "ANTI_ALIASING")]
    pub aa: usize,

    #[clap(short, long, default_value_t = 1000, name = "MAX_ITERATIONS")]
    pub max_itr: usize,

    #[clap(short, long, default_value_t = 20.0)]
    pub bail: f64,

    #[clap(short, long, default_value_t = 1.0)]
    pub speed: f64,

    #[clap(short = 'A', long, default_value_t = 1.0, name = "ACCELERATION")]
    pub acc: f64,

    #[clap(short, long="palette", name="PALETTE")]
    pub palette_path: Option<String>,

    #[clap(long, name="COLOUR_ALGORITHM", conflicts_with = "PALETTE")]
    pub clr_algo: Option<ColourAlgo>,

    #[clap(short='i', long, default_value_t = Interpolation::None, arg_enum)]
    pub interp: Interpolation,

    #[clap(short = 'I', long, default_value = "#000000")]
    pub inside: palette::Srgb<u8>,

    #[clap(short, long, default_value_t = 1)]
    pub frames: usize,

    #[clap(short = 't', long, default_value_t)]
    pub start_time: f64,

    #[clap(short, long)]
    pub julia: bool,

    #[clap(long = "julia_x", default_value_t)]
    pub julia_ctr_x: f64,

    #[clap(long = "julia_y", default_value_t)]
    pub julia_ctr_y: f64,

    #[clap(short = 'T', long)]
    pub threads: Option<usize>,
}

/*
impl Default for SettingsBuilder {
    fn default() -> Self {
        SettingsBuilder {
            width: 640,
            height: 640,
            frames: 1,

            start_time: 0.0,

            aa: 1,
            max_itr: 1000,
            bail: 20.0,

            zoom: 1.0,
            degrees: 0.0,
            ctr_x: 0.0,
            ctr_y: 0.0,

            julia: false,
            julia_ctr_x: 0.0,
            julia_ctr_y: 0.0,

            palette_path: None,
            clr_algo: Some(ColourAlgo::BW),
            interp: Interpolation::None,
            inside: palette::named::BLACK,
            speed: 1.0,
            acc: 1.0,

            threads: None,
        }
    }
}
*/

impl SettingsBuilder {
    pub fn build(&self) -> Settings {
        let width = self.width;
        let width_f64 = width as f64;

        let height = self.height;
        let height_f64 = height as f64;

        let smaller = width.min(height);
        let smaller_f64 = smaller as f64;

        let frame_area = width * height;

        let frames = self.frames;
        let frames_f64 = frames as f64;

        let total_area = frame_area * frames;

        let start_t = self.start_time;

        let aa = self.aa;
        let aa_f64 = aa as f64;

        let aa_sq = aa * aa;
        let aa_sq_f64 = aa_sq as f64;

        let max_itr = self.max_itr;

        let bail = self.bail;
        let bail_sq = bail * bail;
        let bail_ln = bail.ln();

        let zoom = self.zoom;
        let rotation = {
            let radians = self.degrees / 360.0 * std::f64::consts::TAU;
            Complex::new(radians.cos(), radians.sin())
        };
        let center = Complex::new(self.ctr_x, self.ctr_y);

        let julia = self.julia;
        let julia_ctr = Complex::new(self.julia_ctr_x, self.julia_ctr_y);

        let clr_algo =
            if let Some(path) = self.palette_path.as_ref().cloned() {
                ColourAlgo::Palette(Palette::from(path))
            } else {
                self.clr_algo.clone().unwrap_or(ColourAlgo::BW)
            };
        let interp = self.interp;
        let inside: LinSrgb = *LinSrgb::from_raw(&self.inside.into_raw::<[u8;3]>().map(|c| c as f64 / u8::MAX as f64));
        let speed = self.speed;
        let acc = self.acc;

        let threads = self.threads;

        assert!(width >= 1, "Width must be at least 1");
        assert!(height >= 1, "Height must be at least 1");
        assert!(frames >= 1, "Frames must be at least 1");
        assert!(aa >= 1, "Anti-aliasing level must be at least 1");
        assert!(bail >= 20.0, "Bailout must be at least 20");
        if let ColourAlgo::Bands(ref size) = clr_algo {
            assert!(
                (0.0..=1.0).contains(size),
                "Band size must be between 0 and 1"
            );
        }

        Settings {
            width,
            width_f64,
            height,
            height_f64,
            smaller,
            smaller_f64,
            frame_area,
            frames,
            frames_f64,
            total_area,
            start_t,
            aa,
            aa_f64,
            aa_sq,
            aa_sq_f64,
            max_itr,
            bail,
            bail_sq,
            bail_ln,
            zoom,
            rotation,
            center,
            julia,
            julia_ctr,
            clr_algo,
            interp,
            inside,
            speed,
            acc,
            threads,
        }
    }
}

#[non_exhaustive]
pub struct Settings {
    pub width: usize,
    pub width_f64: f64,

    pub height: usize,
    pub height_f64: f64,

    pub smaller: usize,
    pub smaller_f64: f64,

    pub frame_area: usize,

    pub frames: usize,
    pub frames_f64: f64,

    pub total_area: usize,

    pub start_t: f64,

    pub aa: usize,
    pub aa_f64: f64,

    pub aa_sq: usize,
    pub aa_sq_f64: f64,

    pub max_itr: usize,

    pub bail: f64,
    pub bail_sq: f64,
    pub bail_ln: f64,

    pub zoom: f64,
    pub rotation: Complex<f64>,
    pub center: Complex<f64>,

    pub julia: bool,
    pub julia_ctr: Complex<f64>,

    pub clr_algo: ColourAlgo,
    pub interp: Interpolation,
    pub inside: LinSrgb,
    pub speed: f64,
    pub acc: f64,

    pub threads: Option<usize>,
}
