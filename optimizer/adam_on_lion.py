"""
Based on: https://github.com/lucidrains/lion-pytorch/blob/main/lion_pytorch/lion_pytorch.py
"""

from typing import Tuple, Optional, Callable

import torch
from torch.optim.optimizer import Optimizer


def exists(val):
    return val is not None


class AdamOnLion(Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.999),
        gammas: Tuple[float, float] = (0.9, 0.99),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        use_triton: bool = False
    ):
        assert lr > 0.
        assert all([0. <= beta <= 1. for beta in betas])
        assert all([0. <= gamma <= 1. for gamma in gammas])
        assert eps > 0.

        defaults = dict(
            lr = lr,
            betas = betas,
            gammas = gammas,
            eps = eps,
            weight_decay = weight_decay
        )

        super().__init__(params, defaults)

    def update_fn(self, p, grad, m_lion, m_adam, v_adam, lr, wd, beta1, beta2, gamma1, gamma2, eps, step):
        # stepweight decay
        grad.add_(p, alpha = wd)

        m_adam.mul_(beta1).add_(grad, alpha = 1 - beta1)
        v_adam.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)


        m_ = m_adam.clone().div_(1 - beta1**step)
        v_ = v_adam.clone().div_(1 - beta2**step)

        #v_ = torch.add(v_, 1e-7)
        #v_[v_ == 0.0] = 1e-7

        # weight update
        adam_update = m_.div_(v_.sqrt().add_(eps))
        lion_update = m_lion.clone().mul_(gamma1).add(grad, alpha = 1 - gamma1).sign_()
        magnitude = torch.norm(adam_update) / torch.norm(lion_update)
        update = lion_update.mul_(magnitude)
        p.add_(update, alpha = -lr)

        # decay the momentum running average coefficient
        m_lion.mul_(gamma2).add_(grad, alpha = 1 - gamma2)

    @torch.no_grad()
    def step(
        self,
        closure: Optional[Callable] = None
    ):

        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in filter(lambda p: exists(p.grad), group['params']):

                grad, lr, wd, gamma1, gamma2, beta1, beta2, eps, state = p.grad, group['lr'], group['weight_decay'], *group['gammas'], *group['betas'], group['eps'], self.state[p]

                # init state - exponential moving average of gradient values

                if len(state) == 0:
                    state['m_lion'] = torch.zeros_like(p)
                    state['m_adam'] = torch.zeros_like(p)
                    state['v_adam'] = torch.zeros_like(p)
                    state['step'] = 1

                m_lion = state['m_lion']
                m_adam = state['m_adam']
                v_adam = state['v_adam']
                step = state['step']

                self.update_fn(
                    p,
                    grad,
                    m_lion,
                    m_adam,
                    v_adam,
                    lr,
                    wd,
                    beta1,
                    beta2,
                    gamma1,
                    gamma2,
                    eps,
                    step
                )

                state['step'] += 1

        return loss