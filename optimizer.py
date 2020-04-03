import torch


def get_grad_var(layers):
    params_grad = []
    for layer in layers:
        for param in layer.parameters():
            if param.requires_grad:
                params_grad.append(param)
    return params_grad


def get_optimizer(model, lrs, idxs=[]):
    childs = list(model.children())

    if len(idxs) == 0:
        params = get_grad_var(childs)
        opt = torch.optim.Adam(params, lr=lrs[0])
        return opt

    layer_groups = []
    last_idx = 0
    for idx in idxs:
        layer_groups.append(childs[last_idx:idx])
        last_idx = idx

    opt_params = zip(layer_groups, lrs)
    opt = torch.optim.Adam([{'params': get_grad_var(p[0]), 'lr': p[1]} for p in opt_params])
    return opt