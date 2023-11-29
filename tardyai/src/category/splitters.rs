use rand::seq::SliceRandom;

pub trait Splitter<T> {
    fn split(&mut self, files: Vec<T>) -> (Vec<T>, Vec<T>, Vec<T>);
}

pub struct RatioSplitter {
    rng: rand::rngs::StdRng,
    validation: f32,
    test: f32,
}

impl RatioSplitter {
    pub fn with_seed_validation_test(seed: u64, validation: f32, test: f32) -> Self {
        assert!(validation + test < 1.0);
        assert!(validation >= 0.0);
        assert!(test >= 0.0);
        let rng = rand::SeedableRng::seed_from_u64(seed);
        Self {
            rng,
            validation,
            test,
        }
    }

    pub fn with_seed_validation(seed: u64, validation: f32) -> Self {
        Self::with_seed_validation_test(seed, validation, 0.0)
    }

    pub fn with_seed(seed: u64) -> Self {
        Self::with_seed_validation_test(seed, 0.2, 0.0)
    }
}

impl Default for RatioSplitter {
    fn default() -> Self {
        Self::with_seed(0)
    }
}

impl<T: Ord> Splitter<T> for RatioSplitter {
    fn split(&mut self, mut files: Vec<T>) -> (Vec<T>, Vec<T>, Vec<T>) {
        files.sort();
        files.shuffle(&mut self.rng);

        let validation = (files.len() as f32 * self.validation) as usize;
        let test = (files.len() as f32 * self.test) as usize;

        let validation: Vec<T> = files.drain(..validation).collect();
        let test: Vec<T> = files.drain(..test).collect();
        let training: Vec<T> = files;

        (training, validation, test)
    }
}
