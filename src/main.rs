use std::{
    env,
    fmt::Debug,
    fs::{ self, File },
    io::{ Write, BufRead, BufReader },
    sync::{ mpsc, Arc },
    str::FromStr,
    path::Path,
    collections::BTreeMap,
};

use png::{BitDepth, ColorType, Encoder};
use num::Complex;
use threadpool::ThreadPool;

use std::f64::consts::E;
use std::f64::consts::TAU;
use std::f64::consts::PI;

type Pixel = [u8; 3];
type Colour = [f64; 3];

type EscapeTime = Option<f64>;

mod matrix {
    pub struct Matrix<T> {
        width: usize,
        height: usize,
        #[allow(dead_code)]
        area: usize,
        vec: Vec<T>,
    }

    impl<T> Matrix<T> {
        pub fn from_vec(vec: Vec<T>, width: usize) -> Self {
            let height = vec.len() / width;
            assert!(width == vec.len() / height);
            Matrix { width, height, area: width * height, vec }
        }

        pub fn width(&self) -> usize {
            self.width
        }

        pub fn height(&self) -> usize {
            self.height
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

    impl<T> From<Matrix<T>> for Vec<T> {
        fn from(matrix: Matrix<T>) -> Self {
            matrix.vec
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

        pub fn square(side: usize) -> Self {
            Matrix::with_default(side, side, Default::default())
        }
    }
}

use matrix::Matrix;

#[allow(dead_code)]
struct Renderer {
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

    threads:   usize,
}

impl Renderer {
    fn new() -> Self {
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
            inside      : [0.0; 3],
            speed       : 1.0,
            acc         : 1.0,

            threads     : 1,
        }
    }

    // builder functions
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

    fn threads(&mut self, threads: usize) -> &mut Self {
        self.threads = threads;
        self
    }

    fn render(self) {
        let rndr = Arc::new(self);
        match rndr.frames {
            1 => rndr.render_image(),
            _ => rndr.render_anim()
        }
    }

    fn render_image(self: Arc<Self>) {
        let file = File::create("mandelbrot.png").expect("Couldn't open file");
        let mut encoder = Encoder::new(file, self.width as u32, self.height as u32);
        encoder.set_color(ColorType::RGB);
        encoder.set_depth(BitDepth::Eight);
        let mut writer = encoder.write_header().unwrap().into_stream_writer();

        let pool = ThreadPool::new(self.threads);
        let (tx, rx) = mpsc::channel();

        for y in 0..self.height {
            let tx = tx.clone();
            let rndr = self.clone();
            pool.execute(move || {
                for x in 0..rndr.width {
                    let p = get_pixel(x, y, rndr.start_t, &rndr);
                    tx.send((rndr.width * y + x, p)).unwrap();
                }
            });
        }

        // need to drop original tx, or rx iterator will never end
        drop(tx);

        let mut count = 0;
        let mut buffer: BTreeMap<usize, Pixel> = BTreeMap::new();
        let mut waiting_for = 0;
        for msg in rx {
            let (idx, p) = msg;

            if idx == waiting_for {
                let mut p: &Pixel = &p;
                loop {
                    count += 1;
                    writer.write(p).unwrap();
                    waiting_for += 1;
                    if let Some(new_p) = buffer.get(&waiting_for) {
                        p = new_p;
                    } else {
                        break;
                    }
                }
            } else {
                buffer.insert(idx, p);
            }

            print!("\r{}% rendered", count * 100 / self.width / self.height);
            std::io::stdout().flush().unwrap();
        }
    }

    fn render_anim(self: Arc<Self>) {
        assert_ne!(self.frames, 1);
        let pool = ThreadPool::new(self.threads);

        let framepath = Path::new("frames");
        if framepath.exists() {
            fs::remove_dir_all(framepath).unwrap();
        }
        fs::create_dir(framepath).unwrap();

        let escapes: Arc<Matrix<EscapeTime>> = Arc::new(calc_escapes(&self, &pool));

        for frame in 0..self.frames {
            // t is in the range [0, 1)
            let t = frame as f64 / self.frames_f;

            let image_data: Vec<u8> = colourize(&escapes, t, &self, &pool);

            let file = File::create(format!("{}/{}.png", framepath.display(), frame)).expect("Couldn't open file");

            let mut encoder = Encoder::new(file, self.width as u32, self.height as u32);
            encoder.set_color(ColorType::RGB);
            encoder.set_depth(BitDepth::Eight);

            let mut writer = encoder.write_header().unwrap();
            writer
                .write_image_data(&image_data)
                .expect("Couldn't write to file");

            print!("\r{}/{} frames", frame + 1, self.frames);
            std::io::stdout().flush().unwrap();
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
    let mut rndr = Renderer::new();

    {
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
        setting!(threads);

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
    }

    rndr.render();

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
    Palette(Vec<Colour>),
    //Iq([[f64; 3]; 4]),
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

fn get_colour(escape: &EscapeTime, t: f64, rndr: &Renderer) -> Colour {
    if let Some(escape) = escape {
        let escape = escape.powf(rndr.acc) * rndr.speed;

        match &rndr.clr_algo {
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

                match &rndr.interp {
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
        rndr.inside
    }
}

// x and y are start offsets for getting escapes from matrix
fn calc_aa(x: usize, y: usize, t: f64, rndr: &Renderer, escapes: &Matrix<EscapeTime>) -> Pixel {
    let mut sum: Colour = [0.0; 3];
    for yaa in 0..rndr.aa {
        for xaa in 0..rndr.aa {
            let sample = get_colour(escapes.get(x * rndr.aa + xaa, y * rndr.aa + yaa).unwrap(), t, rndr);
            for i in 0..3 {
                sum[i] += sample[i];
            }
        }
    }
    for c in &mut sum {
        *c /= rndr.aa_f * rndr.aa_f;
    }
    let mut pix: Pixel = Default::default();
    for (pix_c, sum_c) in pix.iter_mut().zip(sum.iter()) {
        *pix_c = (sum_c.powf(1.0 / 2.2) * 255.0) as u8;
    }
    pix
}

fn get_pixel(x: usize, y: usize, t: f64, rndr: &Renderer) -> Pixel {
    let mut escapes: Matrix<EscapeTime> = Matrix::square(rndr.aa);
    let x = x as f64;
    let y = y as f64;
    for yaa in 0..rndr.aa {
        for xaa in 0..rndr.aa {
            let c = image_to_complex(x + xaa as f64 / rndr.aa_f, y + yaa as f64 / rndr.aa_f, rndr);
            *escapes.get_mut(xaa, yaa).unwrap() = calc_at(&c, rndr);
        }
    }
    calc_aa(0, 0, t, rndr, &escapes)
}

fn deg_to_rad(deg: f64) -> f64 {
    deg / 360.0 * TAU
}

fn rotate_complex(c: &Complex<f64>, angle: f64, origin: &Complex<f64>) -> Complex<f64> {
    let angle = deg_to_rad(angle);
    (c - origin) * Complex::new(angle.cos(), angle.sin()) + origin
}

// take a point in image coordinates, and return its location in the complex plane
fn image_to_complex(x: f64, y: f64, rndr: &Renderer) -> Complex<f64> {
    let c = (Complex::new(x, y) - Complex::new(rndr.width_f, rndr.height_f) / 2.0) / rndr.smaller_f * 4.0 / rndr.zoom;
    if rndr.julia {
        rotate_complex(&c, rndr.angle, &Complex::new(0.0, 0.0)) + rndr.ctr_julia
    } else {
        c + rndr.center
    }
}

// calculate the escape time at a point c in the complex plane
// if c is in the Mandelbrot set, returns None
fn calc_at(c: &Complex<f64>, rndr: &Renderer) -> EscapeTime {
    let mut z = c.clone();
    let mut itr = 1;

    loop {
        z = z * z + if rndr.julia { &rndr.center } else { c };

        if z.norm_sqr() > rndr.bail_sq {
            let itr = itr as f64;
            return Some(itr - (z.norm().log(E) / rndr.bail.log(E)).log(2.0));
        }

        itr += 1;

        if itr >= rndr.max_itr {
            return None;
        }
    }
}

fn calc_escapes(rndr: &Arc<Renderer>, pool: &ThreadPool) -> Matrix<EscapeTime> {
    let (tx, rx) = mpsc::channel();

    let mut escapes: Matrix<EscapeTime> = Matrix::new(rndr.width * rndr.aa, rndr.height * rndr.aa);
    let width = escapes.width();

    for y in 0 .. rndr.height * rndr.aa {
        let tx = tx.clone();
        let rndr = Arc::clone(rndr);

        pool.execute(move || {
            let mut row: Vec<EscapeTime> = Vec::with_capacity(width);
            for x in 0 .. rndr.width * rndr.aa {
                let c = image_to_complex(x as f64 / rndr.aa_f, y as f64 / rndr.aa_f, &rndr);
                row.push(calc_at(&c, &rndr));
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
        for (x, esc) in row.iter().enumerate() {
            *escapes.get_mut(x, y).unwrap() = *esc;
        }
        count += 1;

        print!("\r{}% calculated", count * 100 / rndr.height / rndr.aa);
        std::io::stdout().flush().unwrap();
    }

    println!();

    escapes
}

fn colourize(escapes: &Arc<Matrix<EscapeTime>>, t: f64, rndr: &Arc<Renderer>, pool: &ThreadPool) -> Vec<u8> {
    let (tx, rx) = mpsc::channel();

    let mut data: Matrix<Pixel> = Matrix::new(rndr.width, rndr.height);

    for y in 0..rndr.height {
        let tx = tx.clone();
        let rndr = Arc::clone(rndr);
        let escapes = Arc::clone(escapes);

        pool.execute(move || {
            tx.send((y, (0..rndr.width).map(|x| calc_aa(x,y,t,&rndr,&escapes)).collect::<Vec<Pixel>>())).unwrap();
        });
    }

    // need to drop original tx, or rx iterator will never end
    drop(tx);

    let mut count = 0;
    for msg in rx {
        let (y, row) = msg;
        for (x, pix) in row.into_iter().enumerate() {
            *data.get_mut(x, y).unwrap() = pix;
        }
        count += 1;

        if rndr.frames == 1 {
            print!("\r{}% colourized", count * 100 / rndr.height);
            std::io::stdout().flush().unwrap();
        }
    }

    if rndr.frames == 1 {
        println!();
    }

    Vec::from(data).into_iter().flatten().collect()
}
