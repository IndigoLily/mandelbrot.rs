use std::sync::Arc;
use std::thread::{self, sleep, JoinHandle};
use std::time::Duration;
use std::io::Write;
use atomic_counter::{RelaxedCounter, AtomicCounter};

#[derive(Debug)]
pub struct Progress {
    count: Arc<RelaxedCounter>,
    jh: JoinHandle<()>,
}

const WAIT: Duration = Duration::from_millis(1000 / 24);

impl Progress {
    pub fn new(process: &str, total: usize) -> Self {
        let process = process.to_owned();
        let count = Arc::new(RelaxedCounter::new(0));

        let jh = thread::spawn({
            let counter = Arc::clone(&count);
            move || {
                let mut count = 0;
                while count < total {
                    print!("\r{}% {}", count * 100 / total, process);
                    std::io::stdout().flush().unwrap();
                    sleep(WAIT);
                    count = counter.get();
                }
                println!("\r100% {}", process);
            }
        });

        Progress { count, jh }
    }

    pub fn inc(&self) {
        self.count.inc();
    }

    pub fn join(self) {
        self.jh.join().unwrap();
    }
}
