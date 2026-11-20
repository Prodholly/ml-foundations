# Day 8 — Full Task: Mastering Neural Networks from NumPy to Advanced PyTorch

**Dataset:** MNIST for phases 1–2 (clean, fast, 70k samples, well-understood baseline).  
**Dataset:** CIFAR-10 for phase 3 (adds spatial structure, color channels, harder generalization — exercises every advanced PyTorch tool).  
**Why these two:** MNIST gives you a tight feedback loop while you are learning mechanics. CIFAR-10 forces you to care about initialization, schedulers, mixed precision, and regularization because the naive approach fails.

---

## Phase 1 — MLP in Pure NumPy (No ML Libraries)

**Goal:** Understand every equation that runs inside a neural network. No PyTorch, no autograd, no shortcuts. When you finish this phase you should be able to explain the full forward and backward pass on a whiteboard.

### 1.1 — Data Pipeline
**File:** `numpy_mlp/data.py`

- Load MNIST from `torchvision.datasets.MNIST` (download only — do not use any PyTorch tensor ops)
- Convert to NumPy arrays immediately: `X_train (60000, 784)`, `y_train (60000,)`
- Normalize: subtract mean (computed on train only), divide by std — shape annotations required
- One-hot encode labels: `Y_train shape (60000, 10)`
- Write a `batch_generator(X, Y, batch_size, shuffle=True)` that yields `(X_batch, Y_batch)` indefinitely
- Split: 80/10/10 train/val/test from the 60k training set; use the 10k test set as held-out final test

**Tests:** `tests/test_data.py`
- batches have correct shape
- normalization mean ≈ 0, std ≈ 1 on train set
- one-hot rows sum to 1

### 1.2 — Activation Functions
**File:** `numpy_mlp/activations.py`

Implement each from its equation with both forward and backward (gradient of the activation):

| Function | Forward | Backward (dL/dx given dL/dy) |
|---|---|---|
| ReLU | max(0, x) | dL/dy · 1(x > 0) |
| GELU | x · Φ(x) approx | numerical or exact |
| Swish | x · σ(x) | x·σ(x)·(1−σ(x)) + σ(x) |
| Softmax | exp(x)/Σexp | (covered in loss) |

For each: plot the function and its derivative across [-3, 3] → `figures/activations.png`  
Note where gradients die (ReLU for x < 0) and where they do not (GELU, Swish).

**Tests:** `tests/test_activations.py`
- relu(0) == 0, relu(-1) == 0, relu(1) == 1
- gradient check each activation with centered finite difference (reuse Day 6's checker)

### 1.3 — Weight Initialization
**File:** `numpy_mlp/init_schemes.py`

| Scheme | Formula | Best with |
|---|---|---|
| Random Gaussian | N(0, 0.01) | nothing — baseline to show it fails |
| Xavier / Glorot | N(0, 2/(fan_in + fan_out)) | sigmoid, tanh |
| He / Kaiming | N(0, 2/fan_in) | ReLU |
| LeCun | N(0, 1/fan_in) | SELU |

Experiment: train 5 epochs with each init, log gradient norms per layer per epoch, plot → `figures/init_comparison.png`  
You should see vanishing gradients with Gaussian init and healthy norms with He.

**Tests:** `tests/test_init.py`
- He init variance ≈ 2/fan_in (within 10%)
- Xavier init variance ≈ 2/(fan_in + fan_out) (within 10%)

### 1.4 — Loss Functions
**File:** `numpy_mlp/losses.py`

- Cross-entropy: `L = -1/N * sum(Y_true * log(Y_pred + 1e-9))` — shape (N, C) → scalar
- Gradient of combined softmax + cross-entropy: `dL/dz = (softmax(z) - Y_true) / N` — derive this by hand first, then implement
- MSE (bonus): for regression practice

**Tests:** `tests/test_losses.py`
- cross_entropy of perfect predictions → near 0
- cross_entropy of uniform predictions → log(10) ≈ 2.303
- gradient check: numerical grad vs analytical grad < 1e-5

### 1.5 — Linear Layer
**File:** `numpy_mlp/layers.py`

Implement a `LinearLayer` class with:
- `__init__(in_features, out_features, init_scheme)` — store W, b
- `forward(X)` — compute Z = XW^T + b, cache X for backward, return Z — shape annotations required
- `backward(dL_dZ)` — compute dL_dX, dL_dW, dL_db; return dL_dX; store dL_dW and dL_db as attributes

Full derivation required as comments:
```
# Z = X @ W.T + b     shape: (N, out)
# dL/dW = dL/dZ.T @ X  shape: (out, in)
# dL/db = sum(dL/dZ, axis=0)  shape: (out,)
# dL/dX = dL/dZ @ W   shape: (N, in)
```

**Tests:** `tests/test_layers.py`
- gradient check: numerical vs analytical for W, b, X — all < 1e-5
- verify shapes for all outputs

### 1.6 — MLP Class
**File:** `numpy_mlp/model.py`

```python
class NumpyMLP:
    def __init__(self, layer_sizes, activation, init_scheme):
        # e.g. layer_sizes = [784, 256, 128, 10]
        # stores list of LinearLayer + list of activation functions
    
    def forward(self, X):
        # apply layers in sequence, cache activations for backward
    
    def backward(self, dL_dZ_last):
        # walk backward through layers and activations
    
    def parameters(self):
        # yields (W, b, dW, db) for each layer
    
    def zero_grad(self):
        # zero all dW and db
```

### 1.7 — Optimizers in NumPy
**File:** `numpy_mlp/optimizers.py`

Implement each from its update equation:

**SGD with momentum:**
```
v = beta * v - lr * grad
theta = theta + v
```

**Adam:**
```
m = beta1 * m + (1 - beta1) * g
v = beta2 * v + (1 - beta2) * g**2
m_hat = m / (1 - beta1**t)
v_hat = v / (1 - beta2**t)
theta -= lr * m_hat / (sqrt(v_hat) + eps)
```

**AdamW:** Adam but weight decay on theta directly, not through gradient:
```
theta *= (1 - lr * weight_decay)   # before the Adam step
```

**Tests:** `tests/test_optimizers.py`
- one step of SGD updates parameter in correct direction
- Adam bias correction at t=1 is applied
- AdamW weight decay shrinks weights independently of gradient

### 1.8 — Regularization
**File:** `numpy_mlp/regularization.py`

- **L2:** add `lambda * sum(W**2)` to loss; add `2 * lambda * W` to dL/dW
- **L1:** add `lambda * sum(|W|)` to loss; add `lambda * sign(W)` to dL/dW
- **Dropout:** during forward, sample `mask = Bernoulli(1-p)`, scale by `1/(1-p)`, apply; during eval, pass through unchanged — the mask must be cached for backward

**Tests:** `tests/test_regularization.py`
- dropout mask has correct fraction of zeros
- L2 grad matches numerical grad with regularization included

### 1.9 — Training Loop
**File:** `numpy_mlp/trainer.py`

```python
def train(model, optimizer, X_train, Y_train, X_val, Y_val,
          epochs, batch_size, patience):
    # returns: train_losses, val_losses, train_accs, val_accs
    # early stopping on val_loss with patience
    # logging: epoch, train_loss, val_loss, train_acc, val_acc
```

Plot at end: loss curves + accuracy curves → `figures/numpy_training_curves.png`

### 1.10 — Full Run and Benchmark
**File:** `numpy_mlp/run.py`

- Train He-init + ReLU + Adam + L2 on MNIST
- Target: **>97% test accuracy**
- Log wall time per epoch
- Final evaluation on held-out test set
- Print confusion matrix

**Commit message for Phase 1:**
```
phase 1: MLP from scratch in NumPy — >97% MNIST

- data pipeline with normalization and batching
- ReLU/GELU/Swish with analytical gradients
- Xavier/He/LeCun init comparison
- LinearLayer with full backward pass + gradient check
- SGD momentum + Adam + AdamW from equations
- L1/L2 regularization + dropout
- training loop with early stopping
- 97%+ test accuracy, all tests passing
```

---

## Phase 2 — MLP in PyTorch Without nn.Module

**Goal:** Use PyTorch tensors and autograd but own every parameter yourself. No nn.Linear, no nn.Module, no nn.functional for layers. You manage the graph, the update, and the grad zeroing. This phase proves you understand what nn.Module abstracts away.

### 2.1 — Tensor-Based Linear Layer
**File:** `torch_raw/layers.py`

```python
class RawLinear:
    def __init__(self, in_features, out_features, device):
        # W = torch.randn(..., requires_grad=True, device=device)
        # b = torch.zeros(..., requires_grad=True, device=device)
    
    def forward(self, x):
        # return x @ self.W.T + self.b   # shape annotations required
    
    def parameters(self):
        return [self.W, self.b]
    
    def zero_grad(self):
        # W.grad = None; b.grad = None
```

### 2.2 — Autograd Verification
**File:** `torch_raw/verify_autograd.py`

For the same RawLinear layer:
- Compute backward manually (as in Phase 1)
- Let autograd compute backward
- Assert they match within 1e-5 on the same input
- This is the key proof that your Phase 1 math is correct

### 2.3 — Raw MLP Class
**File:** `torch_raw/model.py`

Same structure as Phase 1 `NumpyMLP` but:
- All tensors are torch.Tensor with requires_grad=True
- Forward uses torch tensor ops (no autograd's forward functions — just @ and +)
- Backward is called via `loss.backward()` — autograd fills .grad
- Update parameters manually: `with torch.no_grad(): W -= lr * W.grad`

### 2.4 — Raw Optimizers
**File:** `torch_raw/optimizers.py`

Implement SGD + momentum, Adam, AdamW using only torch.Tensor ops (no torch.optim).  
They must hold internal state (momentum buffers, m/v buffers) across steps.

### 2.5 — Dropout and Batch Norm (Tensor-Level)
**File:** `torch_raw/regularization.py`

- Dropout: `torch.bernoulli(torch.full(x.shape, 1-p))` — scale by 1/(1-p) in train mode
- Batch Norm: running mean and running var as plain tensors (no nn.BatchNorm1d)
  - forward train: normalize with batch stats, update running stats with momentum
  - forward eval: normalize with running stats

### 2.6 — Training Loop with Full Instrumentation
**File:** `torch_raw/trainer.py`

Add on top of Phase 1 trainer:
- `register_hook` on each weight to log gradient norms per layer per batch → store for plotting
- Plot gradient norm heatmap per layer per epoch → `figures/grad_norms_raw.png`
- Log GPU memory if CUDA available
- Early stopping, train/val/test split, loss + accuracy curves → `figures/torch_raw_training.png`

### 2.7 — torchviz Graph
**File:** `torch_raw/visualize_graph.py`

Use `torchviz.make_dot(loss, params={...})` to render the computational graph for a 2-layer MLP on a small batch → `figures/compute_graph.png`

### 2.8 — Full Run
- Target: **>97% MNIST** matching Phase 1 — if the numbers differ significantly, your optimizers diverge
- Benchmark Phase 1 vs Phase 2 wall time — record both in a comment in `torch_raw/run.py`

**Commit message for Phase 2:**
```
phase 2: raw PyTorch tensors, no nn.Module — autograd verified

- RawLinear layer matching Phase 1 math
- autograd vs manual backward verified <1e-5 on same input
- SGD/Adam/AdamW in pure tensor ops
- dropout + batch norm at tensor level
- gradient norm logging + compute graph visualization
- same 97%+ MNIST accuracy as Phase 1
```

---

## Phase 3 — Full PyTorch with nn.Module: Advanced to Expert

**Dataset switches to CIFAR-10** (3×32×32 RGB images, 10 classes, 50k train / 10k test).

**Goal:** Use every significant corner of the PyTorch library. By the end of this phase you should be able to implement any paper's architecture and training recipe.

### 3.1 — Dataset and Transforms
**File:** `torch_nn/data.py`

```python
# Training transforms
transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                         std=[0.2023, 0.1994, 0.2010]),
])
# Val/test transforms: ToTensor + Normalize only
```

Custom `CIFARDataset` subclassing `torch.utils.data.Dataset`:
- `__len__`, `__getitem__`, `__repr__`
- `WeightedRandomSampler` for class-balanced sampling
- `DataLoader` with `pin_memory=True`, `num_workers=4`, `persistent_workers=True`

**Tests:** `tests/test_data_torch.py`
- image tensor in range [-3, 3] after normalization
- DataLoader yields correct batch shapes `(B, 3, 32, 32)`
- no data leakage between splits (index sets disjoint)

### 3.2 — nn.Module MLP
**File:** `torch_nn/models/mlp.py`

```python
class MLP(nn.Module):
    def __init__(self, in_features, hidden_sizes, out_features,
                 activation, dropout_p, use_batchnorm):
        super().__init__()
        # build with nn.Sequential or explicit layers
        # register all with self.layers = nn.ModuleList([...])
    
    def forward(self, x):
        # x shape: (B, C, H, W) → flatten to (B, C*H*W) first
```

Demonstrate these nn.Module patterns:
- `nn.ModuleList` vs `nn.Sequential` — when to use each
- `register_buffer` for a running stat that is not a parameter
- `named_parameters()`, `named_modules()`, `children()`
- `model.train()` vs `model.eval()` — what actually changes
- `model.to(device)`, `model.half()`, `model.float()`

**Tests:** `tests/test_models.py`
- output shape is `(B, 10)` for any valid batch
- parameter count matches expected formula
- `model.eval()` produces same output twice (no dropout randomness)

### 3.3 — All Activation Functions via nn.functional
**File:** `torch_nn/models/activations_nn.py`

Implement the same activations as Phase 1 but now using `nn.functional`:
- `F.relu`, `F.gelu`, `F.silu` (Swish), `F.leaky_relu`, `F.elu`
- Register as a custom `nn.Module` subclass for each so they can live in `nn.Sequential`
- Plot all activations and their PyTorch-computed gradients side by side → `figures/nn_activations.png`

### 3.4 — All Initialization Schemes via nn.init
**File:** `torch_nn/models/init_nn.py`

```python
def initialize_model(model, scheme):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            if scheme == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight)
            elif scheme == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif scheme == 'orthogonal':
                nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)
```

Compare gradient flow across schemes in first 5 epochs on CIFAR-10 → `figures/init_nn_comparison.png`

### 3.5 — All Optimizers
**File:** `torch_nn/training/optimizers_nn.py`

Use `torch.optim` for all of these — but write a comparison harness that trains the same model with each:

| Optimizer | Key hyperparams |
|---|---|
| SGD + momentum | lr=0.1, momentum=0.9, nesterov=True |
| Adam | lr=1e-3, betas=(0.9, 0.999), eps=1e-8 |
| AdamW | lr=1e-3, weight_decay=0.01 |
| RMSprop | lr=1e-3, alpha=0.99 |
| LBFGS (bonus) | lr=1, max_iter=20 |

Plot all loss curves on same axes → `figures/optimizer_comparison.png`

### 3.6 — All LR Schedulers
**File:** `torch_nn/training/schedulers_nn.py`

| Scheduler | Use case |
|---|---|
| `StepLR(step_size=30, gamma=0.1)` | Simple decay every N epochs |
| `CosineAnnealingLR(T_max=epochs)` | Smooth decay, no restarts |
| `CosineAnnealingWarmRestarts(T_0=10)` | Multiple restarts for escaping minima |
| `OneCycleLR(max_lr, pct_start=0.3)` | Fast training in fewer epochs |
| Custom warmup + cosine | Standard in Transformer training |

For custom warmup + cosine, implement using `LambdaLR`:
```python
def warmup_cosine(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return step / warmup_steps
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * progress))
scheduler = LambdaLR(optimizer, lr_lambda=warmup_cosine)
```

Plot all LR schedules over 100 epochs → `figures/scheduler_comparison.png`

### 3.7 — Regularization: Full Suite
**File:** `torch_nn/training/regularization_nn.py`

- `nn.Dropout(p=0.5)` — verify only active in train mode
- `nn.BatchNorm1d` — understand running_mean, running_var, num_batches_tracked
- `nn.LayerNorm` — when to use over BatchNorm (batch-size independent)
- `nn.Dropout2d` — for CNN features (drops entire channels)
- L1 / L2 manual penalty added to loss
- Gradient clipping: `torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)`

**Tests:** `tests/test_regularization_nn.py`
- dropout inactive in eval mode
- batchnorm running stats update during train, freeze during eval
- grad clip works: after clip, `max(p.grad.norm() for p in model.parameters()) <= max_norm`

### 3.8 — Mixed Precision Training
**File:** `torch_nn/training/amp_nn.py`

```python
scaler = torch.cuda.amp.GradScaler()

for X, y in loader:
    optimizer.zero_grad()
    with torch.autocast(device_type='cuda', dtype=torch.float16):
        logits = model(X)      # shape: (B, 10)
        loss = criterion(logits, y)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
```

Understand what `GradScaler` does: multiplies loss by a scale factor to prevent fp16 underflow, unscales before the optimizer step, reduces scale if inf/nan is detected.  
Benchmark fp32 vs fp16 wall time and GPU memory on CIFAR-10 → record in `amp_nn.py` header comment.

### 3.9 — Checkpointing and Resumption
**File:** `torch_nn/training/checkpointing_nn.py`

Save everything needed to resume exactly:
```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'best_val_loss': best_val_loss,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'config': config,
}
torch.save(checkpoint, path)
```

Verify resumption: train 5 epochs, save, load, train 5 more — final loss must match training without interruption.

### 3.10 — Reproducibility
**File:** `torch_nn/training/reproducibility_nn.py`

```python
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
```

Verify: run twice with same seed → identical loss curves. Run with different seeds → different but stable curves.

### 3.11 — Debugging and Profiling
**File:** `torch_nn/debugging/`

**Hooks (`hooks.py`):**
```python
# Forward hook: log activation statistics
def fwd_hook(module, input, output):
    print(f"{module.__class__.__name__}: mean={output.mean():.3f}, std={output.std():.3f}")

# Backward hook: log gradient statistics  
def bwd_hook(module, grad_input, grad_output):
    print(f"{module.__class__.__name__} grad: norm={grad_output[0].norm():.3f}")

handle = layer.register_forward_hook(fwd_hook)
handle = layer.register_full_backward_hook(bwd_hook)
# always: handle.remove() when done
```

**Gradient checking (`grad_check.py`):**  
Reuse Day 6's numerical gradient checker on a full nn.Module. Use `torch.autograd.gradcheck` as a second oracle.

**Anomaly detection:**
```python
torch.autograd.set_detect_anomaly(True)
# run forward + backward — will raise RuntimeError with stack trace on NaN/Inf
```

**Profiling (`profile_nn.py`):**
```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
    profile_memory=True,
    with_stack=True,
) as prof:
    model(X)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

### 3.12 — Loss Landscape Visualization
**File:** `torch_nn/debugging/loss_landscape.py`

Filter normalization method (from "Visualizing the Loss Landscape of Neural Nets", Li et al. 2018):
```python
# Pick two random direction vectors, normalize per-filter
# Perturb weights: theta_perturbed = theta + alpha * d1 + beta * d2
# Evaluate loss on a fixed batch for each (alpha, beta) on a grid
# Plot 2D surface and contour map
```

→ `figures/loss_landscape_nn.png`

### 3.13 — Custom Autograd Function
**File:** `torch_nn/advanced/custom_autograd.py`

Implement a custom op using `torch.autograd.Function`:
```python
class ClippedReLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clip_val):
        ctx.save_for_backward(x)
        ctx.clip_val = clip_val
        return x.clamp(0, clip_val)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        grad = grad_output * ((x > 0) & (x < ctx.clip_val)).float()
        return grad, None
```

Verify with `torch.autograd.gradcheck`.

### 3.14 — Custom Loss Functions
**File:** `torch_nn/advanced/custom_losses.py`

Implement:
- Label smoothing cross-entropy (used in almost all modern training)
- Focal loss (for class imbalance)
- Each as an `nn.Module` subclass

```python
class LabelSmoothingCE(nn.Module):
    def forward(self, logits, targets):
        # logits: (B, C), targets: (B,)
        # smooth_targets: (B, C) where non-target classes get eps/C
```

### 3.15 — Full Training Framework
**File:** `torch_nn/training/trainer_nn.py`

Bring everything together:
```python
class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion,
                 scaler, device, config):
    
    def train_epoch(self, loader) -> dict  # returns metrics
    def val_epoch(self, loader) -> dict
    def fit(self, train_loader, val_loader) -> History
    def evaluate(self, test_loader) -> dict
    def save_checkpoint(self, path)
    def load_checkpoint(self, path)
```

### 3.16 — Final Benchmark on CIFAR-10
**File:** `torch_nn/run_cifar10.py`

Best config: He init + ReLU + AdamW (lr=1e-3, wd=0.01) + OneCycleLR + dropout(0.3) + batchnorm + mixed precision + grad clipping  
Target: **>85% test accuracy on CIFAR-10**

Produce final plots:
- `figures/cifar10_training_curves.png` — loss + accuracy over epochs
- `figures/cifar10_confusion_matrix.png`
- `figures/cifar10_loss_landscape.png`

**Commit message for Phase 3:**
```
phase 3: full pytorch nn.Module mastery — >85% CIFAR-10

- custom Dataset + DataLoader with augmentation
- nn.Module MLP with every registration pattern demonstrated
- all torch.optim optimizers compared
- all LR schedulers including custom warmup+cosine
- full regularization suite (dropout, batchnorm, layernorm, grad clip)
- mixed precision with GradScaler
- complete checkpointing and reproducibility
- forward/backward hooks + anomaly detection + profiler
- loss landscape visualization
- custom autograd.Function + custom loss functions
- reusable Trainer class
- >85% CIFAR-10 test accuracy, all tests passing
```

---

## Project Structure

```
8_pytorch/
├── numpy_mlp/                   Phase 1: pure NumPy
│   ├── __init__.py
│   ├── data.py
│   ├── activations.py
│   ├── init_schemes.py
│   ├── losses.py
│   ├── layers.py
│   ├── model.py
│   ├── optimizers.py
│   ├── regularization.py
│   ├── trainer.py
│   └── run.py
│
├── torch_raw/                   Phase 2: PyTorch tensors, no nn.Module
│   ├── __init__.py
│   ├── layers.py
│   ├── model.py
│   ├── optimizers.py
│   ├── regularization.py
│   ├── trainer.py
│   ├── verify_autograd.py
│   ├── visualize_graph.py
│   └── run.py
│
├── torch_nn/                    Phase 3: full nn.Module mastery
│   ├── __init__.py
│   ├── data.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── mlp.py
│   │   ├── activations_nn.py
│   │   └── init_nn.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── optimizers_nn.py
│   │   ├── schedulers_nn.py
│   │   ├── regularization_nn.py
│   │   ├── amp_nn.py
│   │   ├── checkpointing_nn.py
│   │   ├── reproducibility_nn.py
│   │   └── trainer_nn.py
│   ├── debugging/
│   │   ├── __init__.py
│   │   ├── hooks.py
│   │   ├── grad_check.py
│   │   └── loss_landscape.py
│   ├── advanced/
│   │   ├── __init__.py
│   │   ├── custom_autograd.py
│   │   └── custom_losses.py
│   └── run_cifar10.py
│
├── tests/
│   ├── test_data.py
│   ├── test_activations.py
│   ├── test_init.py
│   ├── test_losses.py
│   ├── test_layers.py
│   ├── test_optimizers.py
│   ├── test_regularization.py
│   ├── test_data_torch.py
│   ├── test_models.py
│   └── test_regularization_nn.py
│
├── figures/
│   ├── activations.png
│   ├── init_comparison.png
│   ├── numpy_training_curves.png
│   ├── grad_norms_raw.png
│   ├── compute_graph.png
│   ├── nn_activations.png
│   ├── init_nn_comparison.png
│   ├── optimizer_comparison.png
│   ├── scheduler_comparison.png
│   ├── loss_landscape.png
│   ├── cifar10_training_curves.png
│   ├── cifar10_confusion_matrix.png
│   └── cifar10_loss_landscape.png
│
└── pyproject.toml
```

**CLI to create all directories and files:**
```bash
mkdir -p 8_pytorch/numpy_mlp
mkdir -p 8_pytorch/torch_raw
mkdir -p 8_pytorch/torch_nn/models
mkdir -p 8_pytorch/torch_nn/training
mkdir -p 8_pytorch/torch_nn/debugging
mkdir -p 8_pytorch/torch_nn/advanced
mkdir -p 8_pytorch/tests
mkdir -p 8_pytorch/figures

touch 8_pytorch/numpy_mlp/__init__.py 8_pytorch/numpy_mlp/data.py 8_pytorch/numpy_mlp/activations.py 8_pytorch/numpy_mlp/init_schemes.py 8_pytorch/numpy_mlp/losses.py 8_pytorch/numpy_mlp/layers.py 8_pytorch/numpy_mlp/model.py 8_pytorch/numpy_mlp/optimizers.py 8_pytorch/numpy_mlp/regularization.py 8_pytorch/numpy_mlp/trainer.py 8_pytorch/numpy_mlp/run.py

touch 8_pytorch/torch_raw/__init__.py 8_pytorch/torch_raw/layers.py 8_pytorch/torch_raw/model.py 8_pytorch/torch_raw/optimizers.py 8_pytorch/torch_raw/regularization.py 8_pytorch/torch_raw/trainer.py 8_pytorch/torch_raw/verify_autograd.py 8_pytorch/torch_raw/visualize_graph.py 8_pytorch/torch_raw/run.py

touch 8_pytorch/torch_nn/__init__.py 8_pytorch/torch_nn/data.py 8_pytorch/torch_nn/run_cifar10.py
touch 8_pytorch/torch_nn/models/__init__.py 8_pytorch/torch_nn/models/mlp.py 8_pytorch/torch_nn/models/activations_nn.py 8_pytorch/torch_nn/models/init_nn.py
touch 8_pytorch/torch_nn/training/__init__.py 8_pytorch/torch_nn/training/optimizers_nn.py 8_pytorch/torch_nn/training/schedulers_nn.py 8_pytorch/torch_nn/training/regularization_nn.py 8_pytorch/torch_nn/training/amp_nn.py 8_pytorch/torch_nn/training/checkpointing_nn.py 8_pytorch/torch_nn/training/reproducibility_nn.py 8_pytorch/torch_nn/training/trainer_nn.py
touch 8_pytorch/torch_nn/debugging/__init__.py 8_pytorch/torch_nn/debugging/hooks.py 8_pytorch/torch_nn/debugging/grad_check.py 8_pytorch/torch_nn/debugging/loss_landscape.py
touch 8_pytorch/torch_nn/advanced/__init__.py 8_pytorch/torch_nn/advanced/custom_autograd.py 8_pytorch/torch_nn/advanced/custom_losses.py

touch 8_pytorch/tests/test_data.py 8_pytorch/tests/test_activations.py 8_pytorch/tests/test_init.py 8_pytorch/tests/test_losses.py 8_pytorch/tests/test_layers.py 8_pytorch/tests/test_optimizers.py 8_pytorch/tests/test_regularization.py 8_pytorch/tests/test_data_torch.py 8_pytorch/tests/test_models.py 8_pytorch/tests/test_regularization_nn.py
```

**Add to root pyproject.toml:**
```toml
# testpaths: add "8_pytorch/tests"
# pythonpath: add "8_pytorch"
```

---

## Mastery Checklist

Before calling this day done, you should be able to answer every one of these without looking anything up:

**Phase 1:**
- [ ] Derive dL/dW for a linear layer from scratch on a whiteboard
- [ ] Explain why He init uses 2/fan_in and Xavier uses 2/(fan_in + fan_out)
- [ ] Why does the combined softmax + cross-entropy gradient simplify to (p - y)/N?
- [ ] What does the Adam bias correction term (1 - beta^t) do and why is it needed at t=1?
- [ ] Why does AdamW apply weight decay to the parameter directly rather than through the gradient?

**Phase 2:**
- [ ] What does requires_grad=True do — what is actually tracked in the graph?
- [ ] Why do you need torch.no_grad() when manually updating parameters?
- [ ] What does retain_graph=True do — when do you need it?
- [ ] What does .detach() do — when do you need it?

**Phase 3:**
- [ ] What is the difference between nn.Parameter and register_buffer?
- [ ] Why does model.eval() affect Dropout and BatchNorm but not Linear?
- [ ] What does GradScaler do, and why is unscale_() needed before gradient clipping?
- [ ] What is filter normalization in the loss landscape, and why is it needed for fair comparison?
- [ ] What is the difference between torch.autograd.Function and a regular nn.Module?
