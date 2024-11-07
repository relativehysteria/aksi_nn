pub type Operation = (OpType, [usize; 2]);

#[derive(Debug, Clone)]
pub enum OpType {
    Add,
    Sub,
    Mul,
    Div,
    Tanh,
    Pow,
    Exp,
}

#[derive(Debug, Clone)]
pub struct Value {
    data: f64,
    op: Option<Operation>,
    grad: f64,
}

impl Value {
    pub fn new(data: f64, op: Operation) -> Self {
        Self {
            data,
            op: Some(op),
            grad: 0.0,
        }
    }

    pub fn new_const(data: f64) -> Self {
        Self {
            data,
            op: None,
            grad: 0.0,
        }
    }
}

#[derive(Debug)]
pub struct Context {
    values: Vec<Value>,
}

impl Context {
    pub fn new() -> Self {
        Self {
            values: Vec::new(),
        }
    }

    pub fn push(&mut self, val: f64) -> usize {
        self.push_val(Value::new_const(val))
    }

    pub fn push_val(&mut self, val: Value) -> usize {
        let idx = self.values.len();
        self.values.push(val);
        idx
    }

    pub fn value(&self, idx: usize) -> f64 {
        self.values[idx].data
    }

    pub fn grad(&self, idx: usize) -> f64 {
        self.values[idx].grad
    }

    fn apply_op<F>(&mut self, idx1: usize, idx2: usize,
                   op_type: OpType, op: F) -> usize
    where
        F: Fn(f64, f64) -> f64,
    {
        let result = op(self.values[idx1].data, self.values[idx2].data);
        self.push_val(Value::new(result, (op_type, [idx1, idx2])))
    }

    pub fn add(&mut self, idx1: usize, idx2: usize) -> usize {
        self.apply_op(idx1, idx2, OpType::Add, |a, b| a + b)
    }

    pub fn sub(&mut self, idx1: usize, idx2: usize) -> usize {
        self.apply_op(idx1, idx2, OpType::Sub, |a, b| a - b)
    }

    pub fn mul(&mut self, idx1: usize, idx2: usize) -> usize {
        self.apply_op(idx1, idx2, OpType::Mul, |a, b| a * b)
    }

    pub fn div(&mut self, idx1: usize, idx2: usize) -> usize {
        self.apply_op(idx1, idx2, OpType::Div, |a, b| a / b)
    }

    pub fn pow(&mut self, base_idx: usize, exponent_idx: usize) -> usize {
        self.apply_op(base_idx, exponent_idx, OpType::Pow, |a, b| a.powf(b))
    }

    pub fn exp(&mut self, idx: usize) -> usize {
        self.apply_op(idx, 0, OpType::Exp, |a, _| a.exp())
    }

    pub fn tanh(&mut self, idx: usize) -> usize {
        self.apply_op(idx, 0, OpType::Tanh, |a, _| {
            let x = core::f64::consts::E.powf(2.0 * a) - 1.0;
            let y = core::f64::consts::E.powf(2.0 * a) + 1.0;
            x / y
        })
    }

    pub fn backprop(&mut self, output_idx: usize) {
        // Reset all gradients to zero, except for the output node
        self.values.iter_mut().for_each(|v| v.grad = 0.0);
        self.values[output_idx].grad = 1.0;

        // Traverse nodes in reverse to accumulate gradients
        for idx in (0..=output_idx).rev() {
            let val = self.values[idx].clone();

            // Constants/inputs need no adjustments
            if val.op.is_none() { continue; }

            let (optype, operands) = val.op.unwrap();
            match optype {
                OpType::Add => {
                    // d(output)/d(a) = 1
                    // d(output)/d(b) = 1
                    for &op in &operands {
                        self.values[op].grad += self.values[idx].grad;
                    }
                },
                OpType::Sub => {
                    // d(output)/d(a) = 1
                    // d(output)/d(b) = -1
                    let idx_a = operands[0];
                    let idx_b = operands[1];
                    self.values[idx_a].grad += self.values[idx].grad;
                    self.values[idx_b].grad -= self.values[idx].grad;
                }
                OpType::Mul => {
                    // d(output)/d(a) = b
                    // d(output)/d(b) = a
                    let idx_a = operands[0];
                    let idx_b = operands[1];
                    let a = self.values[idx_a].data;
                    let b = self.values[idx_b].data;
                    self.values[idx_a].grad += b * self.values[idx].grad;
                    self.values[idx_b].grad += a * self.values[idx].grad;
                },
                OpType::Div => {
                    // d(output)/d(a) = 1 / b
                    // d(output)/d(b) = -a / b^2
                    let idx_a = operands[0];
                    let idx_b = operands[1];
                    let a = self.values[idx_a].data;
                    let b = self.values[idx_b].data;
                    self.values[idx_a].grad +=
                        (1.0 / b) * self.values[idx].grad;
                    self.values[idx_b].grad -=
                        (a / (b * b)) * self.values[idx].grad;
                },
                OpType::Pow => {
                    // d(output)/d(a) = b * a^(b-1)
                    // d(output)/d(b) = a^b * ln(a)
                    let idx_a = operands[0];
                    let idx_b = operands[1];
                    let a = self.values[idx_a].data;
                    let b = self.values[idx_b].data;
                    self.values[idx_a].grad +=
                        b * a.powf(b - 1.0) * self.values[idx].grad;
                    self.values[idx_b].grad +=
                        a.powf(b) * self.values[idx].grad * a.ln();
                },
                OpType::Tanh => {
                    // d(output)/d(x) = 1 - tanh(x)^2
                    self.values[operands[0]].grad +=
                        (1.0 - val.data.powi(2)) * self.values[idx].grad;
                },
                OpType::Exp => {
                    // d(output)/d(x) = exp(x)
                    let idx_a = operands[0];
                    let a = self.values[idx_a].data;
                    self.values[idx_a].grad += a.exp() * self.values[idx].grad;
                },
            }
        }
    }
}
