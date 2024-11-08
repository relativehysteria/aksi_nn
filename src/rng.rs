/// A deterministic PRNG (xorshift)
pub struct Rng(usize);

impl Rng {
    /// Creates a new RNG
    pub fn new(seed: usize) -> Self {
        let mut rng = Self(seed);

        // First couple of runs to avoid bad seeds
        (0..64).for_each(|_| { rng.rand(); });

        rng
    }

    /// Returns a pseudo-random (predetermined) number in the range [0.0, 1.0]
    pub fn rand(&mut self) -> f64 {
        let ret = self.0;
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 17;
        self.0 ^= self.0 << 43;
        ret as f64 / usize::MAX as f64
    }

    /// Returns a pseudo-random (predetermined) number within a given range
    pub fn range(&mut self, min: f64, max: f64) -> f64 {
        let scale = max - min;
        (self.rand() * scale) + min
    }
}
