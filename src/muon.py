import torch
import torch.optim as optim

def newton_schulz5(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to orthogonalize a matrix G.
    Approximates the nearest orthogonal matrix (polar factorization).
    """
    assert G.ndim == 2
    # Coefficients for the 5th order Newton-Schulz iteration
    a, b, c = (3.4445, -4.7750, 2.0315)
    
    X = G.to(dtype=torch.bfloat16)
    X /= (X.norm() + eps)
    
    # Ensure we work with the smaller dimension for efficiency
    transposed = False
    if X.size(0) > X.size(1):
        X = X.T
        transposed = True
        
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if transposed:
        X = X.T
        
    return X.to(dtype=G.dtype)

class Muon(optim.Optimizer):
    """
    Muon: MomentUm Orthogonalized by Newton-Schulz.
    Specifically designed for 2D parameters (hidden weights).
    """
    def __init__(self, params, lr=1e-3, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['momentum_buffer'] = torch.zeros_like(p)

                buf = state['momentum_buffer']
                buf.mul_(group['momentum']).add_(grad)

                if group['nesterov']:
                    d_p = grad.add(buf, alpha=group['momentum'])
                else:
                    d_p = buf

                # Apply weight decay
                if group['weight_decay'] != 0:
                    p.mul_(1 - group['lr'] * group['weight_decay'])

                # Orthogonalize the update
                original_shape = d_p.shape
                if d_p.ndim > 2:
                    d_p = d_p.view(original_shape[0], -1)
                
                update = newton_schulz5(d_p, steps=group['ns_steps'])
                
                # Standard accurate scaling: sqrt(max_dim)
                max_dim = max(d_p.size(0), d_p.size(1))
                scale = max_dim**0.5
                
                p.add_(update.view(original_shape), alpha=-group['lr'] * scale)

        return loss

class MuonWithAuxAdam(optim.Optimizer):
    """
    A wrapper that uses Muon for 2D parameters and AdamW for others.
    """
    def __init__(self, model, lr=1e-3, adam_lr=3e-4, weight_decay=0.01, **muon_kwargs):
        muon_params = []
        adam_params = []
        
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            
            # Transformer backbone weights (2D) and FlowHead weights (2D) use Muon
            # Embeddings, norms, custom projectors, and LoRA adapters use AdamW
            is_backbone = "layers" in name
            is_head = "flow_head" in name
            is_excluded = any(x in name for x in ["embed", "norm", "proj", "lora"])
            
            if p.ndim >= 2 and (is_backbone or is_head) and not is_excluded:
                muon_params.append(p)
            else:
                adam_params.append(p)
        
        self.muon = Muon(muon_params, lr=lr, weight_decay=weight_decay, **muon_kwargs)
        self.adam = optim.AdamW(adam_params, lr=adam_lr, weight_decay=weight_decay)
        
        # Super init to register hooks and state
        super().__init__(muon_params + adam_params, {})
        self.param_groups = self.muon.param_groups + self.adam.param_groups

    def step(self, closure=None):
        loss = self.muon.step(closure)
        self.adam.step()
        return loss

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {
            "muon": self.muon.state_dict(),
            "adam": self.adam.state_dict()
        }

    def load_state_dict(self, state_dict):
        self.muon.load_state_dict(state_dict["muon"])
        self.adam.load_state_dict(state_dict["adam"])
