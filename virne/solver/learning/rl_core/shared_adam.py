# ==============================================================================
# Copyright 2023 GeminiLight (wtfly2018@gmail.com). All Rights Reserved.
# ==============================================================================


import torch


class SharedAdam(torch.optim.Adam):
    """
    Shared optimizer, the parameters in the optimizer will shared in the multiprocessors.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
                 weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg']
                state['exp_avg_sq']

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

    @staticmethod
    def from_optim(optim):
        param_groups_list = []
        for optim_group in optim.param_groups:
            param_groups_list.append({
                'params': optim_group['params'],
                'lr': optim_group['lr'],
                'betas': optim_group['betas'],
                'eps': optim_group['eps'],
                'weight_decay': optim_group['weight_decay'],
            })
        return SharedAdam(param_groups_list)


def sync_gradients(shared_model, local_model):
    for (shared_name, shared_param), (local_name, local_param) in zip(shared_model.named_parameters(), local_model.named_parameters()):
        try:
            shared_param._grad = local_param.grad.clone().to(shared_param.device)
        except:
            # print('Warning: No gradient!', shared_name, type(shared_param), local_name, type(local_param))
            pass




    # local_params = local_model.parameters()
    # for shared_param in shared_model.parameters():
    #     local_grad = next(local_params)._grad
    #     print(local_grad)
    #     shared_param._grad = local_grad if not shared_param.grad else shared_param._grad