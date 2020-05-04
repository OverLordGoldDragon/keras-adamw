# -*- coding: utf-8 -*-
import random
import numpy as np
import tensorflow as tf
from termcolor import colored


WARN = colored('WARNING:', 'red')


def _init_weight_decays(init_fn):
    def get_and_zero_decays(cls, model=None, **kwargs):
        zero_penalties = kwargs.pop('zero_penalties', True)
        weight_decays = kwargs.pop('weight_decays', None)
        if not zero_penalties:
            print(WARN, "loss-based weight penalties should be set to zero. "
                  "(set `zero_penalties=True`)")

        if weight_decays is not None and model is not None:
            print(WARN, "`weight_decays` is set automatically when "
                  "passing in `model`; will override supplied")
        if model is not None:
            weight_decays = get_weight_decays(model, zero_penalties)

        init_fn(cls, weight_decays=weight_decays, **kwargs)
    return get_and_zero_decays


def get_weight_decays(model, zero_penalties=False, verbose=1):
    wd_dict = {}
    for layer in model.layers:
        layer_penalties = _get_layer_penalties(layer, zero_penalties)
        if layer_penalties:
            for p in layer_penalties:
                weight_name, weight_penalty = p
                if not all(wp == 0 for wp in weight_penalty):
                    wd_dict.update({weight_name: weight_penalty})
    return wd_dict


def fill_dict_in_order(_dict, values_list):
    for idx, key in enumerate(_dict.keys()):
        _dict[key] = values_list[idx]
    return _dict


def _get_layer_penalties(layer, zero_penalties=False):
    if hasattr(layer, 'cell') or \
      (hasattr(layer, 'layer') and hasattr(layer.layer, 'cell')):
        return _rnn_decays(layer, zero_penalties)
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


def _rnn_decays(layer, zero_penalties=False):
    penalties = []
    if hasattr(layer, 'backward_layer'):
        for layer in [layer.forward_layer, layer.backward_layer]:
            penalties += _cell_l2regs(layer.cell, zero_penalties)
        return penalties
    else:
        return _cell_l2regs(layer.cell, zero_penalties)


def _cell_l2regs(rnn_cell, zero_penalties=False):
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
