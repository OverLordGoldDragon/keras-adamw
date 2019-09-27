'''Helper methods for keras_adamw.py

'''


def get_weight_decays(model, verbose=1):
    wd_dict = {}
    for layer in model.layers:
        layer_l2regs = _get_layer_l2regs(layer)
        if layer_l2regs:
            for layer_l2 in layer_l2regs:
                weight_name, weight_l2 = layer_l2
                wd_dict.update({weight_name: weight_l2})
                if weight_l2 != 0 and verbose:
                    print(("WARNING: {} l2-regularization = {} - should be "
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
    _layer = layer.layer if 'backward_layer' in layer.__dict__ else layer
    cell = _layer.cell

    l2_lambda_krb = []
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
