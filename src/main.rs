use core::num::NonZero;
use neurnet::{Rng, Context, CtxIdx, MultiLayerPerceptron as MLP};

fn rdtsc() -> usize {
    unsafe { core::arch::x86_64::_rdtsc() as usize }
}

fn main() {
    let inputs = vec![
        vec![2.0,  3.0, -1.0],
        vec![3.0, -1.0,  0.5],
        vec![0.5,  1.0,  1.0],
        vec![1.0,  1.0, -1.0],
    ];

    let targets = vec![1.0, -1.0, -1.0, 1.0];
    let learning_rate = 0.1;
    let epochs = 20;

    let mut ctx = Context::new();
    let mut rng = Rng::new(rdtsc());

    let mlp = MLP::new(
        &mut ctx,
        &mut rng,
        &[NonZero::new(3).unwrap(),
          NonZero::new(4).unwrap(),
          NonZero::new(4).unwrap(),
          NonZero::new(1).unwrap()]);

    for epoch in 0..epochs {
        // Store loss terms for each sample
        let mut loss_terms = vec![];

        // Forward pass: calculate predictions and loss for each input-target pair
        for (input, &target) in inputs.iter().zip(targets.iter()) {
            // Push each input component to the Context
            let indices: Vec<CtxIdx> =
                input.iter().map(|&i| ctx.push(i)).collect();

            // Forward pass through the MLP
            let pred_idx = mlp.forward(&mut ctx, &indices)[0];

            // Compute squared error (y_pred - y_target)^2
            let target_idx       = ctx.push(target);
            let diff_idx         = ctx.sub(pred_idx, target_idx);
            let squared_diff_idx = ctx.mul(diff_idx, diff_idx);

            // Store the squared difference for later summation
            loss_terms.push(squared_diff_idx);
        }

        // Sum all loss terms to get the total loss
        let total_loss_idx = ctx.sum(&loss_terms);
        let loss_value     = ctx.value(total_loss_idx);

        // Backward pass: reset gradients and perform backpropagation
        mlp.parameters().for_each(|p| ctx.clear_grad(p));
        ctx.backward(total_loss_idx);

        // Update parameters based on gradients
        for param in mlp.parameters() {
            // p.data += -learning_rate * p.grad
            let grad   = ctx.grad(param);
            let update = ctx.value(param) - learning_rate * grad;
            (*ctx.get_mut(param)).data = update;
        }

        // Print epoch number and loss value
        println!("Epoch {:>02}: Loss = {:.4}", epoch, loss_value);
    }
}
