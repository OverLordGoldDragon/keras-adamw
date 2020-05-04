from tensorflow.python.ops import math_ops
'''Helper methods for optimizers_225tf.py optimizers
'''


def _apply_weight_decays(cls, var, var_t):
    wd = cls.weight_decays[var.name]
    wd_normalized = wd * math_ops.sqrt(cls.batch_size / cls.total_iterations_wd)
    wdn_printable = wd * (cls.batch_size / cls.total_iterations_wd) ** (1/2)
    var_t = var_t - cls.eta_t * wd_normalized * var

    if cls.init_verbose and not cls._init_notified:
        print('{} weight decay set for {}'.format(wdn_printable, var.name))
    return var_t


def _compute_eta_t(cls):
    PI = 3.141592653589793
    t_frac = math_ops.cast(cls.t_cur / cls.total_iterations, 'float32')
    eta_t = cls.eta_min + 0.5 * (cls.eta_max - cls.eta_min) * \
        (1 + math_ops.cos(PI * t_frac))
    return eta_t


def _apply_lr_multiplier(cls, lr_t, var):
    multiplier_name = [mult_name for mult_name in cls.lr_multipliers
                       if mult_name in var.name]
    if multiplier_name != []:
        lr_mult = cls.lr_multipliers[multiplier_name[0]]
    else:
        lr_mult = 1
    lr_t = lr_t * lr_mult

    if cls.init_verbose and not cls._init_notified:
        lr_t_printable = cls._init_lr * lr_mult
        if lr_mult != 1:
            print('{} init learning rate set for {} -- {}'.format(
               '%.e' % lr_t_printable, var.name, lr_t))
        else:
            print('No change in learning rate {} -- {}'.format(var.name,
                  lr_t_printable))
    return lr_t
