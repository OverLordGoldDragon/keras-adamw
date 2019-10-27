from termcolor import colored
import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow.python.ops import math_ops
import random
'''Helper methods for TensorFlow < 2, Keras < 2.3.0
   tf.keras.optimizer optimizers
'''


def warn_str():
    return colored('WARNING: ', 'red')


def get_weight_decays(model, verbose=1):
    wd_dict = {}
    for layer in model.layers:
        layer_l2regs = _get_layer_l2regs(layer)
        if layer_l2regs:
            for layer_l2 in layer_l2regs:
                weight_name, weight_l2 = layer_l2
                wd_dict.update({weight_name: weight_l2})
                if weight_l2 != 0 and verbose:
                    print((warn_str() + "{} l2-regularization = {} - should be "
                          "set 0 before compiling model").format(
                                  weight_name, weight_l2))
    return wd_dict


def fill_dict_in_order(_dict, _list_of_vals):
    for idx, key in enumerate(_dict.keys()):
        _dict[key] = _list_of_vals[idx]
    return _dict


def _get_layer_l2regs(layer):
    if hasattr(layer, 'layer') or hasattr(layer, 'cell'):
        return _rnn_l2regs(layer)
    else:
        l2_lambda_kb = []
        for weight_name in ['kernel', 'bias']:
            _lambda = getattr(layer, weight_name + '_regularizer', None)
            if _lambda is not None:
                l2_lambda_kb.append([getattr(layer, weight_name).name,
                                     float(_lambda.l2)])
        return l2_lambda_kb


def _rnn_l2regs(layer):
    l2_lambda_krb = []
    if hasattr(layer, 'backward_layer'):
        for layer in [layer.forward_layer, layer.backward_layer]:
            l2_lambda_krb += _cell_l2regs(layer.cell)
        return l2_lambda_krb
    else:
        return _cell_l2regs(layer.cell)


def _cell_l2regs(rnn_cell):
    cell = rnn_cell
    l2_lambda_krb = []  # kernel-recurrent-bias

    if hasattr(cell, 'kernel_regularizer') or \
       hasattr(cell, 'recurrent_regularizer') or hasattr(cell, 'bias_regularizer'):
        for weight_name in ['kernel', 'recurrent', 'bias']:
            _lambda = getattr(cell, weight_name + '_regularizer', None)
            if _lambda is not None:
                weight_name = weight_name if 'recurrent' not in weight_name \
                                          else 'recurrent_kernel'
                l2_lambda_krb.append([getattr(cell, weight_name).name,
                                      float(_lambda.l2)])
    return l2_lambda_krb


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


def _check_args(total_iterations, use_cosine_annealing, weight_decays):
    if use_cosine_annealing and total_iterations != 0:
        print('Using cosine annealing learning rates')
    elif (use_cosine_annealing or weight_decays != {}) and total_iterations == 0:
        print(warn_str() + "'total_iterations'==0, must be !=0 to use "
              + "cosine annealing and/or weight decays; "
              + "proceeding without either")


def reset_seeds(reset_graph_with_backend=None, verbose=1):
    if reset_graph_with_backend is not None:
        K = reset_graph_with_backend
        K.clear_session()
        tf.compat.v1.reset_default_graph()
        if verbose:
            print("KERAS AND TENSORFLOW GRAPHS RESET")

    np.random.seed(1)
    random.seed(2)
    if tf.__version__[0] == '2':
        tf.random.set_seed(3)
    else:
        tf.set_random_seed(3)
    if verbose:
        print("RANDOM SEEDS RESET")


def K_eval(x, backend=K):
    K = backend
    try:
        return K.get_value(K.to_dense(x))
    except Exception as e:
        try:
            eval_fn = K.function([], [x])
            return eval_fn([])[0]
        except Exception as e:
            return K.eval(x)
