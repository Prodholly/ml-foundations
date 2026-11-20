import torch

class RawLinear:
    def __init__(self, in_features, out_features, device, dtype=torch.float32):
        self.in_features = in_features
        self.out_features = out_features
        
        self.device = device
        scale = (2 / in_features) ** 0.5
    
        self.W = torch.randn((out_features, in_features), device=device, dtype=dtype) * scale
        self.W.requires_grad_()
        self.b = torch.zeros(out_features, device=device, dtype=dtype)
        self.b.requires_grad_()

    def forward(self, x): 
        
        if x.ndim != 2 or x.shape[1] != self.in_features:
            raise ValueError(
                f"i expect input shape (N, {self.in_features}), "
                f"but got {tuple(x.shape)}"
            )
        
        return x @ self.W.T + self.b

    def parameters(self):
        return [self.W, self.b]

    def zero_grad(self):
        self.W.grad = None
        self.b.grad = None

    def __call__(self, x):
        return self.forward(x)
