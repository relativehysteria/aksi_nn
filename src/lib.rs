extern crate alloc;

pub mod value;
pub mod net;
pub mod rng;

pub use value::{Value, Context, CtxIdx};
pub use rng::Rng;
pub use net::{Neuron, Layer, MultiLayerPerceptron};
