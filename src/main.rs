fn main() {
    let dim = 80;
    for y in 0..dim {
        for x in 0..dim * 2 {
            let x = (f64::from(x) / 2.0 - f64::from(dim) / 2.0) / f64::from(dim * 1) * 4.0;
            let y = (f64::from(y)       - f64::from(dim) / 2.0) / f64::from(dim * 1) * 4.0;
            print!("{}", if calc(x, y) { "@" } else { " " });
        }
        print!("\n");
    }
}

fn calc(cr: f64, ci: f64) -> bool {
    let c = (cr, ci);
    let mut z = (0.0, 0.0);
    for _n in 0..1_000 {
        iter(&c, &mut z);
        if z.0 * z.0 + z.1 * z.1 > 4.0 {
            return false;
        };
    }
    true
}

fn iter(c: &(f64, f64), z: &mut (f64, f64)) {
    // z^2+c
    *z = (z.0 * z.0 - z.1 * z.1 + c.0, 2.0 * z.0 * z.1 + c.1);
}
