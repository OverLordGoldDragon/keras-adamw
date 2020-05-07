from tensorflow.python.ops import math_ops
'''Helper methods for optimizers_225tf.py optimizers
'''


def _apply_weight_decays(cls, var, var_t):
    l1, l2 = cls.weight_decays[var.name]
    if l1 == 0 and l2 == 0:
        if cls.init_verbose and not cls._init_notified:
            print("Both penalties are 0 for %s, will skip" % var.name)
        return var_t

    norm = math_ops.sqrt(cls.batch_size / cls.total_iterations_wd)
    l1_normalized = l1 * norm
    l2_normalized = l2 * norm

    norm_printable = (cls.batch_size / cls.total_iterations_wd) ** (1 / 2)
    l1n_printable = l1 * norm_printable
    l2n_printable = l2 * norm_printable

    if l1 != 0 and l2 != 0:
        decay = l1_normalized * math_ops.sign(var) + l2_normalized * var
    elif l1 != 0:
        decay = l1_normalized * math_ops.sign(var)
    else:
        decay = l2_normalized * var
    var_t = var_t - cls.eta_t * decay

    if cls.init_verbose and not cls._init_notified:
        decays_str = "{}(L1), {}(L2)".format(l1n_printable, l2n_printable)
        print('{} weight decay set for {}'.format(decays_str, var.name))
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
