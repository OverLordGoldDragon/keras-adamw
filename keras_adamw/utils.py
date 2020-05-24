import random
import numpy as np
import tensorflow as tf
from termcolor import colored
from tensorflow.python.ops import math_ops


WARN = colored('WARNING:', 'red')


def _apply_weight_decays(cls, var, var_t):
    l1, l2 = cls.weight_decays[var.name]
    if l1 == 0 and l2 == 0:
        if cls.init_verbose and not cls._init_notified:
            print("Both penalties are 0 for %s, will skip" % var.name)
        return var_t

    norm = math_ops.cast(math_ops.sqrt(cls.batch_size / cls.total_iterations_wd),
                         'float32')
    l1_normalized = l1 * norm
    l2_normalized = l2 * norm

    if l1 != 0 and l2 != 0:
        decay = l1_normalized * math_ops.sign(var) + l2_normalized * var
    elif l1 != 0:
        decay = l1_normalized * math_ops.sign(var)
    else:
        decay = l2_normalized * var
    var_t = var_t - cls.eta_t * decay

    if cls.init_verbose and not cls._init_notified:
        norm_print = (cls.batch_size / cls.total_iterations_wd) ** (1 / 2)
        l1n_print, l2n_print = l1 * norm_print, l2 * norm_print
        decays_str = "{}(L1), {}(L2)".format(l1n_print, l2n_print)
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
        lr_print = cls._init_lr * lr_mult
        if lr_mult != 1:
            print('{} init learning rate set for {} -- {}'.format(
               '%.e' % round(lr_print, 5), var.name, lr_t))
        else:
            print('No change in learning rate {} -- {}'.format(var.name,
                                                               lr_print))
    return lr_t


def _init_weight_decays(model, zero_penalties, weight_decays):
    if not zero_penalties:
        print(WARN, "loss-based weight penalties should be set to zero. "
              "(set `zero_penalties=True`)")
    if weight_decays is not None and model is not None:
        print(WARN, "`weight_decays` is set automatically when "
              "passing in `model`; will override supplied")
    if model is not None:
        weight_decays = get_weight_decays(model, zero_penalties)
    return weight_decays


def get_weight_decays(model, zero_penalties=False):
    wd_dict = {}
    for layer in model.layers:
        layer_penalties = _get_layer_penalties(layer, zero_penalties)
        if layer_penalties:
            for p in layer_penalties:
                weight_name, weight_penalty = p
                if not all(wp == 0 for wp in weight_penalty):
                    wd_dict.update({weight_name: weight_penalty})
    return wd_dict


def _get_layer_penalties(layer, zero_penalties=False):
    if hasattr(layer, 'cell') or \
      (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
        return _rnn_penalties(layer, zero_penalties)
    elif hasattr(layer, 'layer') and not hasattr(layer.layer, 'cell'):
        layer = layer.layer

    penalties= []
    for weight_name in ['kernel', 'bias']:
        _lambda = getattr(layer, weight_name + '_regularizer', None)
        if _lambda is not None:
            l1l2 = (float(_lambda.l1), float(_lambda.l2))
            penalties.append([getattr(layer, weight_name).name, l1l2])
            if zero_penalties:
                _lambda.l1 = np.array(0., dtype=_lambda.l1.dtype)
                _lambda.l2 = np.array(0., dtype=_lambda.l2.dtype)
    return penalties


def _rnn_penalties(layer, zero_penalties=False):
    penalties = []
    if hasattr(layer, 'backward_layer'):
        for layer in [layer.forward_layer, layer.backward_layer]:
            penalties += _cell_penalties(layer.cell, zero_penalties)
        return penalties
    else:
        return _cell_penalties(layer.cell, zero_penalties)


def _cell_penalties(rnn_cell, zero_penalties=False):
    cell = rnn_cell
    penalties = []  # kernel-recurrent-bias

    for weight_idx, weight_type in enumerate(['kernel', 'recurrent', 'bias']):
        _lambda = getattr(cell, weight_type + '_regularizer', None)
        if _lambda is not None:
            weight_name = cell.weights[weight_idx].name
            l1l2 = (float(_lambda.l1), float(_lambda.l2))
            penalties.append([weight_name, l1l2])
            if zero_penalties:
                _lambda.l1 = np.array(0., dtype=_lambda.l1.dtype)
                _lambda.l2 = np.array(0., dtype=_lambda.l2.dtype)
    return penalties


def _check_args(total_iterations, use_cosine_annealing, weight_decays):
    if use_cosine_annealing and total_iterations != 0:
        print('Using cosine annealing learning rates')
    elif (use_cosine_annealing or weight_decays != {}) and total_iterations == 0:
        print(WARN, "'total_iterations'==0, must be !=0 to use "
              "cosine annealing and/or weight decays; "
              "proceeding without either")


def fill_dict_in_order(_dict, values_list):
    for idx, key in enumerate(_dict.keys()):
        _dict[key] = values_list[idx]
    return _dict


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


def K_eval(x, backend):
    K = backend
    try:
        return K.get_value(K.to_dense(x))
    except Exception:
        try:
            eval_fn = K.function([], [x])
            return eval_fn([])[0]
        except Exception:
            try:
                return K.eager(K.eval)(x)
            except Exception:
                return K.eval(x)
