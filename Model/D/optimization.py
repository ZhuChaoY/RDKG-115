import math
import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


class AdamW(Optimizer):
    """Adam <https://arxiv.org/abs/1711.05101>"""

    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.999), eps = 1e-6,
                       weight_decay = 0.0, correct_bias = True):
        defaults = dict(lr = lr, betas = betas, eps = eps, 
                   weight_decay = weight_decay, correct_bias = correct_bias)
        super().__init__(params, defaults)


    def step(self, closure = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']
                state['step'] += 1
                exp_avg.mul_(beta1).add_(grad, alpha = 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group['eps'])
                step_size = group['lr']
                if group['correct_bias']:  
                    bias_correction1 = 1.0 - beta1 ** state['step']
                    bias_correction2 = 1.0 - beta2 ** state['step']
                    step_size = step_size * math.sqrt(bias_correction2) / \
                                bias_correction1
                p.data.addcdiv_(exp_avg, denom, value = -step_size)
                if group['weight_decay'] > 0.0:
                    p.data.add_(p.data, alpha = \
                                -group['lr'] * group['weight_decay'])
        return loss
    
    
    
def warmup_schedule(optimizer, n_warmup, n_step):
    def lr_lambda(k: int):
        if k < n_warmup:
            return float(k) / float(max(1, n_warmup))
        return max(0.0, float(n_step - k) / float(max(1, n_step - n_warmup)))
    return LambdaLR(optimizer, lr_lambda, -1)