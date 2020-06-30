from torch import optim


def select_optimizer(model, lr, lr_orth, alpha, betas, Tdecay, optimizer_name):
    x = [
        {'params': (param for param in model.parameters()
                    if param is not model.rnncell.log_P
                    and param is not model.rnncell.P
                    and param is not model.rnncell.UppT)},
        {'params': model.rnncell.UppT, 'weight_decay': Tdecay}
    ]
    y = [
        {'params': (param for param in model.rnncell.parameters() if param is model.rnncell.log_P)}
    ]
    if optimizer_name == 'rmsprop':
        optimizer = optim.RMSprop(x, lr=lr, alpha=alpha)
        orthog_optimizer = optim.RMSprop(y, lr=lr_orth, alpha=alpha)
    elif optimizer_name == 'adam':
        optimizer = optim.Adam(x, lr=lr, betas=betas)
        orthog_optimizer = optim.Adam(y, lr=lr_orth, betas=betas)
    return optimizer, orthog_optimizer
