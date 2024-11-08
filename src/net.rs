use core::num::NonZero;
use crate::{Context, CtxIdx, Rng};

#[derive(Debug)]
pub struct Neuron {
    weights: Vec<CtxIdx>,
    bias: CtxIdx,
}

impl Neuron {
    pub fn new(ctx: &mut Context, rng: &mut Rng,
               n_inputs: NonZero<usize>) -> Self {
        let bias = ctx.push(rng.range(-1.0, 1.0));
        let weights = (0..n_inputs.into())
            .map(|_| ctx.push(rng.range(-1.0, 1.0)))
            .collect();

        Self {
            weights,
            bias
        }
    }

    pub fn forward(&self, ctx: &mut Context, x: &[CtxIdx]) -> CtxIdx {
        let mut act = self.bias;
        for (&wi, &xi) in self.weights.iter().zip(x.iter()) {
            let idx = ctx.mul(wi, xi);
            act = ctx.add(act, idx);
        }
        ctx.tanh(act)
    }

    pub fn parameters(&self) -> impl Iterator<Item = CtxIdx> + '_ {
        self.weights.iter().copied().chain(core::iter::once(self.bias))
    }
}

#[derive(Debug)]
pub struct Layer {
    neurons: Vec<Neuron>,
}

impl Layer {
    pub fn new(ctx: &mut Context, rng: &mut Rng,
               n_inputs: NonZero<usize>, n_outputs: NonZero<usize>) -> Self {
        let neurons = (0..n_outputs.into())
            .map(|_| Neuron::new(ctx, rng, n_inputs))
            .collect();

        Self { neurons }
    }

    pub fn forward(&self, ctx: &mut Context, x: &[CtxIdx]) -> Vec<CtxIdx> {
        self.neurons.iter().map(|n| n.forward(ctx, x)).collect()
    }

    pub fn parameters(&self) -> impl Iterator<Item = CtxIdx> + '_ {
        self.neurons.iter().flat_map(|neuron| neuron.parameters())
    }
}

#[derive(Debug)]
pub struct MultiLayerPerceptron {
    layers: Vec<Layer>,
}

impl MultiLayerPerceptron {
    // the sizes of all the layers we want
    pub fn new(ctx: &mut Context, rng: &mut Rng,
               topology: &[NonZero<usize>]) -> Self {

        let layers = topology.windows(2)
            .map(|top| Layer::new(ctx, rng, top[0], top[1]))
            .collect();

        Self { layers }
    }

    pub fn forward(&self, ctx: &mut Context, x: &[CtxIdx]) -> Vec<CtxIdx> {
        self.layers.iter()
            .fold(x.to_vec(), |input, layer| layer.forward(ctx, &input))
    }

    pub fn parameters(&self) -> impl Iterator<Item = CtxIdx> + '_ {
        self.layers.iter().flat_map(|layer| layer.parameters())
    }

    pub fn pretty_print(&self, ctx: &Context) {
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            println!("Layer {layer_idx}:");
            for (neuron_idx, neuron) in layer.neurons.iter().enumerate() {
                println!("    Neuron {neuron_idx}:");
                println!("        Bias: {:>28.25?}", ctx.value(neuron.bias));
                println!("        Weights:");
                for &weight in neuron.weights.iter() {
                    print!("            value: {:>23.20?}", ctx.value(weight));
                    println!(" | gradient: {:>23.20?}", ctx.grad(weight));
                }
            }
        }
    }
}
