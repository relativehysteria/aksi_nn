use neurnet::Context;

fn main() {
    let mut ctx = Context::new();

    // inputs
    let x1 = ctx.push(2.0);
    let x2 = ctx.push(0.0);

    // weights
    let w1 = ctx.push(-3.0);
    let w2 = ctx.push(1.0);

    // bias of the neuron
    let b = ctx.push(6.8813735870195432);

    // x1w1 + x2w2 + b
    let x1w1 = ctx.mul(x1, w1);
    let x2w2 = ctx.mul(x2, w2);
    let xw   = ctx.add(x1w1, x2w2);
    let n = ctx.add(xw, b);
    let o = ctx.tanh(n);

    println!("{:?}", ctx);
}
