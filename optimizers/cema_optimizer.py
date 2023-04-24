from typing import List, Iterator, Optional

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer


class CEMA_Optimizer(Optimizer):

    def __init__(self, binary_params: Iterator[Parameter],
                 alpha: float,
                 gamma: float):
        if alpha is None or alpha <= 0:
            raise ValueError(f"Invalid alpha value: {alpha}. Needs to be bigger than 0")
        if gamma is None or gamma <= 0:
            raise ValueError(f"Invalid gamma value: {gamma}. Needs to be bigger than 0")

        defaults = dict(lr=1.0, alpha=alpha, initial_alpha=alpha, gamma=gamma)
        super().__init__(binary_params, defaults)

        with torch.no_grad():
            for group in self.param_groups:
                for i, p in enumerate(group['params']):
                    p.sign_()

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.

        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        all_possible_flips = []
        for i, group in enumerate(self.param_groups):
            params_with_grad = []
            d_p_list = []

            inertia_buffer_list = []  # role of latent weights
            momentum_buffer_list = []

            # Decay alpha according to the learning rate decay schedule
            decay_factor = group.get("lr")
            alpha = group['initial_alpha'] * decay_factor
            group['alpha'] = alpha

            gamma = group['gamma']

            amount_of_weights = 0
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)

                    amount_of_weights += p.numel()

                    state = self.state[p]
                    if 'momentum_buffer' not in state:
                        momentum_buffer_list.append(torch.zeros_like(p))
                    else:
                        momentum_buffer_list.append(state['momentum_buffer'])

                    if 'inertia_buffer' not in state:
                        inertia_buffer_list.append(torch.zeros_like(p))
                    else:
                        inertia_buffer_list.append(state['inertia_buffer'])

            cascaded_ema(params_with_grad,
                         d_p_list,
                         inertia_buffer_list,
                         momentum_buffer_list,
                         alpha,
                         gamma)

            # update momentum_buffers in state
            for p, momentum_buffer in zip(params_with_grad, momentum_buffer_list):
                state = self.state[p]
                state['momentum_buffer'] = momentum_buffer

            for p, inertia_buffer in zip(params_with_grad, inertia_buffer_list):
                state = self.state[p]
                state['inertia_buffer'] = inertia_buffer


def cascaded_ema(params: List[Tensor],
                 grad_list: List[Tensor],
                 inertia_buffer_list: List[Optional[Tensor]],
                 momentum_buffer_list: List[Optional[Tensor]],
                 alpha: float,
                 gamma: float):
    for i, inertia in enumerate(inertia_buffer_list):
        grad = grad_list[i]
        momentum_buf = momentum_buffer_list[i]

        momentum_buf.mul_(1 - gamma).add_(grad, alpha=gamma)
        inertia.mul_(1 - alpha).add_(momentum_buf, alpha=alpha)

    inertia = torch.cat([m.view(-1) for m in inertia_buffer_list])
    b = torch.cat([b.view(-1) for b in params])

    relative_inertia = inertia.mul(b)

    flip_mask = (relative_inertia > 0)

    i = 0
    for param in params:
        n = param.numel()
        param_flip_mask = flip_mask[i:i + n].reshape_as(param)
        param[param_flip_mask] *= -1

        i += n
