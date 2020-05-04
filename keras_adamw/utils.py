import keras.backend as K
from .utils_common import K_eval as KE
'''Helper methods for optimizers
'''

def K_eval(x):
    return KE(x, K)


def _apply_weight_decays(cls, var, var_t):
    l1, l2 = cls.weight_decays[var.name]
    norm = K.cast(K.sqrt(cls.batch_size / cls.total_iterations_wd), 'float32')
    l1_normalized = l1 * norm
    l2_normalized = l2 * norm
    var_t = var_t - cls.eta_t * (l1_normalized * var +
                                 l2_normalized * K.sign(var))

    if cls.init_verbose and not cls._init_notified:
        decays_str = "{}(L1), {}(L2)".format(K_eval(l1_normalized),
                                             K_eval(l2_normalized))
        print('{} weight decay set for {}'.format(decays_str, var.name))
    return var_t


def _compute_eta_t(cls):
    PI = 3.141592653589793
    t_frac = K.cast(cls.t_cur / cls.total_iterations, 'float32')
    eta_t = cls.eta_min + 0.5 * (cls.eta_max - cls.eta_min) * \
        (1 + K.cos(PI * t_frac))
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
        if lr_mult != 1:
            print('{} init learning rate set for {} -- {}'.format(
               '%.e' % round(K_eval(lr_t), 5), var.name, lr_t))
        else:
            print('No change in learning rate {} -- {}'.format(
                                              var.name, K_eval(lr_t)))
    return lr_t
