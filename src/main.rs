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
        self.real = self.real*self.real - self.imag*self.imag;
        self.imag = 2.0 * tmp_real * self.imag;
        self
    }

    fn mag_sq(&self) -> f64 {
        self.real*self.real + self.imag*self.imag
    }
}

fn calc_at(c: &Complex) -> Option<f64> {
    let mut z = Complex::new(0.0, 0.0);
    let mut itr = 0;
    let max_itr = 100;
    loop {
        z.sq().add(c);
        itr += 1;
        if itr >= max_itr {
            return None;
        } else if z.mag_sq() > 4.0 {
            return Some(f64::from(itr));
        }
    }
}

fn main() {
    let width  = 200;
    let height = 200;
}
