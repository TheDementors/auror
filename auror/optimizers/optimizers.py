from torch.optim import (
    Adadelta as _Adadelta,
    Adagrad as _Adagrad,
    Adam as _Adam,
    AdamW as _AdamW,
    SparseAdam as _SparseAdam,
    Adamax as _Adamax,
    ASGD as _ASGD,
    LBFGS as _LBFGS,
    RMSprop as _RMSprop,
    Rprop as _Rprop,
    SGD as _SGD,
)


def adadelta(lr=1.0, rho=0.9, eps=1e-6, weight_decay=0):
    if not 0.0 <= lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= rho <= 1.0:
        raise ValueError("Invalid rho value: {}".format(rho))
    if not 0.0 <= eps:
        raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= weight_decay:
        raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, rho=rho, eps=eps, weight_decay=weight_decay)

    return _Adadelta, defaults


def adagrad(lr=1e-2, lr_decay=0, weight_decay=0, initial_accumulator_value=0, eps=1e-10):
    if not 0.0 <= lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= lr_decay:
        raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
    if not 0.0 <= weight_decay:
        raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
    if not 0.0 <= initial_accumulator_value:
        raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
    if not 0.0 <= eps:
        raise ValueError("Invalid epsilon value: {}".format(eps))

    defaults = dict(
        lr=lr,
        lr_decay=lr_decay,
        eps=eps,
        weight_decay=weight_decay,
        initial_accumulator_value=initial_accumulator_value,
    )

    return _Adagrad, defaults


def adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0, amsgrad=False):
    if not 0.0 <= lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
        raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    if not 0.0 <= weight_decay:
        raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    return _Adam, defaults


def adam_w(lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, amsgrad=False):
    if not 0.0 <= lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
        raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    if not 0.0 <= weight_decay:
        raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    return _AdamW, defaults


def sparse_adam(lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
    if not 0.0 < lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 < eps:
        raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

    defaults = dict(lr=lr, betas=betas, eps=eps)

    return _SparseAdam, defaults


def adamax(lr=2e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
    if not 0.0 <= lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
        raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= betas[0] < 1.0:
        raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
    if not 0.0 <= betas[1] < 1.0:
        raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
    if not 0.0 <= weight_decay:
        raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

    return _Adamax, defaults


def asgd(lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
    if not 0.0 <= lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= weight_decay:
        raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0, weight_decay=weight_decay)

    return _ASGD, defaults


def lbfgs(
    lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-7, tolerance_change=1e-9, history_size=100, line_search_fn=None
):
    defaults = dict(
        lr=lr,
        max_iter=max_iter,
        max_eval=max_eval,
        tolerance_grad=tolerance_grad,
        tolerance_change=tolerance_change,
        history_size=history_size,
        line_search_fn=line_search_fn,
    )

    return _LBFGS, defaults


def rms_prop(lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0, momentum=0, centered=False):
    if not 0.0 <= lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 <= eps:
        raise ValueError("Invalid epsilon value: {}".format(eps))
    if not 0.0 <= momentum:
        raise ValueError("Invalid momentum value: {}".format(momentum))
    if not 0.0 <= weight_decay:
        raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
    if not 0.0 <= alpha:
        raise ValueError("Invalid alpha value: {}".format(alpha))

    defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps, centered=centered, weight_decay=weight_decay)

    return _RMSprop, defaults


def r_prop(lr=1e-2, etas=(0.5, 1.2), step_sizes=(1e-6, 50)):
    if not 0.0 <= lr:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if not 0.0 < etas[0] < 1.0 < etas[1]:
        raise ValueError("Invalid eta values: {}, {}".format(etas[0], etas[1]))

    defaults = dict(lr=lr, etas=etas, step_sizes=step_sizes)

    return _Rprop, defaults


def sgd(lr, momentum=0, dampening=0, weight_decay=0, nesterov=False):
    if lr < 0.0:
        raise ValueError("Invalid learning rate: {}".format(lr))
    if momentum < 0.0:
        raise ValueError("Invalid momentum value: {}".format(momentum))
    if weight_decay < 0.0:
        raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

    defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    return _SGD, defaults
