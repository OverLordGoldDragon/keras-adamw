import os
import tempfile
import numpy as np
import tensorflow as tf

from time import time
from termcolor import cprint

from . import K
from . import Input, Dense, GRU, Bidirectional, Embedding
from . import Model, load_model
from . import l1, l2, l1_l2
from . import maxnorm
from . import Adam, Nadam, SGD
from keras_adamw import AdamW, NadamW, SGDW
from keras_adamw import get_weight_decays, fill_dict_in_order, reset_seeds
from keras_adamw import K_eval


print("TF version: %s" % tf.__version__)
TF_KERAS = bool(os.environ["TF_KERAS"] == '1')
TF_EAGER = bool(os.environ["TF_EAGER"] == '1')
if TF_EAGER:
    print("TF running eagerly")
else:
    tf.compat.v1.disable_eager_execution()
    print("TF running in graph mode")


def test_main():  # Save/Load, Warm Restarts (w/ cosine annealing)
    for optimizer_name in ['AdamW', 'NadamW', 'SGDW']:
        cprint("<< TESTING {} OPTIMIZER >>".format(optimizer_name), 'blue')
        reset_seeds()

        num_batches, num_epochs = 25, 4
        batch_size, timesteps, num_channels = 16, 8, 4
        batch_shape = (batch_size, timesteps, num_channels)
        total_iterations = num_batches  # due to warm restarts

        model = _make_model(batch_shape, l1_reg=1e-4, l2_reg=1e-4)
        optimizer = _make_optimizer(optimizer_name, model, total_iterations)
        model.compile(optimizer, loss='binary_crossentropy')
        assert _valid_weight_decays(model)

        if hasattr(model, '_make_train_function'):  # graph-mode
            model._make_train_function()  # else K.eval before train may fail

        X, Y = _make_data(num_batches, *batch_shape)
        eta_history = []    # for stop-introspection
        t_cur_history = []  # for stop-introspection

        # eta_t is first applied as-is, and only updated AFTER iteration;
        # setting t_cur does not immediately change eta_t.
        # Thus, t_cur must be reset 1 iteration BEFORE epoch ends
        # (t, e) = (t_cur_history[-1], eta_history[-1])
        # (t, e) = (24, 0)    -> RESET -> (-1, 0...)         [on     epoch end]
        # (t, e) = (23, 0...) -> RESET -> (-1, 0) -> (0, 1)  [before epoch end]
        for epoch in range(num_epochs):
            for batch_num in range(num_batches):
                t_cur_history += [K_eval(model.optimizer.t_cur, K)]
                eta_history += [K_eval(model.optimizer.eta_t, K)]
                model.train_on_batch(X[batch_num], Y[batch_num])
                if batch_num == (num_batches - 2):
                    K.set_value(model.optimizer.t_cur, -1)

        assert _valid_cosine_annealing(eta_history, total_iterations, num_epochs)
        assert model.optimizer.get_config()  # ensure value evaluation won't error
        _test_save_load(model, X, optimizer_name, optimizer)

        # cleanup
        del model, optimizer
        reset_seeds(reset_graph_with_backend=K)

        cprint("\n<< {} MAIN TEST PASSED >>\n".format(optimizer_name), 'green')
    cprint("\n<< ALL MAIN TESTS PASSED >>\n", 'green')


def test_misc():  # tests of non-main features to improve coverage
    for optimizer_name in ['AdamW', 'NadamW', 'SGDW']:
        cprint("<< TESTING {} OPTIMIZER >>".format(optimizer_name), 'blue')
        reset_seeds()

        optimizer_kw = {'total_iterations': 0, 'decay': 1e-3,
                        'amsgrad': optimizer_name == 'AdamW',
                        'nesterov': optimizer_name == 'SGDW'}
        num_batches = 4
        batch_size, timesteps = 16, 8
        batch_shape = (batch_size, timesteps)
        embed_input_dim = 5

        # arbitrarily select SGDW for coverage testing
        l1_reg = 1e-4 if optimizer_name == 'SGDW' else 0
        l2_reg = 1e-4 if optimizer_name != 'SGDW' else 0
        if optimizer_name == 'SGDW':
            optimizer_kw['zero_penalties'] = False
            optimizer_kw['weight_decays'] = {}
            optimizer_kw['total_iterations'] = 2

        model = _make_model(batch_shape,
                            embed_input_dim=embed_input_dim,
                            dense_constraint=1,
                            l1_reg=l1_reg, l2_reg=l2_reg,
                            bidirectional=False, sparse=True)
        optimizer = _make_optimizer(optimizer_name, model, **optimizer_kw)
        model.compile(optimizer, loss='sparse_categorical_crossentropy')
        X, Y = _make_data(num_batches, *batch_shape,
                          embed_input_dim=embed_input_dim, sparse=True)

        for batch_num in range(num_batches):
            model.train_on_batch(X[batch_num], Y[batch_num])

        _test_save_load(model, X, optimizer_name, optimizer)

        # util test
        dc = {'lstm': 0, 'dense': 0}
        fill_dict_in_order(dc, [1e-4, 2e-4])
        AdamW(model=model, zero_penalties=False, total_iterations=2)
        AdamW(model=model, weight_decays={'a': 0})

        opt = AdamW(weight_decays={model.layers[1].weights[0].name: (0, 0)},
                    total_iterations=2)
        model.compile(opt, loss='sparse_categorical_crossentropy')
        model.train_on_batch(X[0], Y[0])

        # cleanup
        del model, optimizer
        reset_seeds(reset_graph_with_backend=K)
        try:
            K_eval('x', K)  # for coverage
        except:
            pass

        cprint("\n<< {} MISC TEST PASSED >>\n".format(optimizer_name), 'green')
    cprint("\n<< ALL MISC TESTS PASSED >>\n", 'green')


def test_control():  # tests losses against original optimizers'
    for optimizer_name in ['AdamW', 'NadamW', 'SGDW']:
        cprint("<< TESTING {} OPTIMIZER >>".format(optimizer_name), 'blue')
        pass_txt = "Control Test Passed"

        if optimizer_name == 'AdamW':
            for amsgrad in [True, False]:
                _test_control(optimizer_name, amsgrad=amsgrad)
                print("\n>> AdamW amsgrad={} {}".format(amsgrad, pass_txt))
        elif optimizer_name == 'NadamW':
            _test_control(optimizer_name)

        elif optimizer_name == 'SGDW':
            for nesterov in [True, False]:
                _test_control(optimizer_name, nesterov=nesterov)
                print("\n>> SGDW nesterov={} {}".format(nesterov, pass_txt))

        o_name = optimizer_name
        cprint("\n<< {} {} >>\n".format(o_name, pass_txt.upper()), 'green')

    cprint("\n<< ALL CONTROL TESTS PASSED >>\n", 'green')


def _test_control(optimizer_name, amsgrad=False, nesterov=False):
    optimizer_kw = dict(total_iterations=0, decay=1e-3,
                        amsgrad=amsgrad, nesterov=nesterov,
                        control_mode=True)
    num_batches = 100
    batch_size, timesteps = 16, 32
    batch_shape = (batch_size, timesteps)
    embed_input_dim = 5

    model_kw = dict(batch_shape=batch_shape, dense_constraint=1,
                    embed_input_dim=embed_input_dim, l1_reg=0, l2_reg=0,
                    bidirectional=False, sparse=True)
    loss_name = 'sparse_categorical_crossentropy'
    reset_seeds(verbose=0)
    X, Y = _make_data(num_batches, *batch_shape,
                      embed_input_dim=embed_input_dim, sparse=True)

    reset_seeds(reset_graph_with_backend=K, verbose=0)
    model_custom = _make_model(**model_kw)
    optimizer_custom = _make_optimizer(optimizer_name, model_custom,
                                       **optimizer_kw)
    model_custom.compile(optimizer_custom, loss=loss_name)
    loss_custom = []  # for introspection
    t0 = time()
    for batch_num in range(num_batches):
        loss_custom += [model_custom.train_on_batch(
                X[batch_num], Y[batch_num])]
    print("model_custom -- %s batches -- time: %.2f sec" % (num_batches,
                                                            time() - t0))

    reset_seeds(reset_graph_with_backend=K, verbose=0)
    model_control = _make_model(**model_kw)
    optimizer_control = _make_optimizer(optimizer_name[:-1], model_control,
                                        **optimizer_kw)
    model_control.compile(optimizer_control, loss=loss_name)
    loss_control = []  # for introspection
    t0 = time()
    for batch_num in range(num_batches):
        loss_control += [model_control.train_on_batch(X[batch_num],
                                                      Y[batch_num])]
    print("model_control -- %s batches -- time: %.2f sec" % (num_batches,
                                                             time() - t0))

    loss_diff = np.abs(np.array(loss_custom) -
                       np.array(loss_control))
    print("%s max loss diff: %e" % (optimizer_name, np.max(loss_diff)))

    assert np.allclose(loss_custom, loss_control, rtol=0, atol=1e-3)
    # cleanup
    del model_custom, model_control
    del optimizer_custom, optimizer_control
    reset_seeds(reset_graph_with_backend=K, verbose=0)


def _test_save_load(model, X, optimizer_name, optimizer):
    saved_model_preds = model.predict(X[0])
    saved_model_weights = K.batch_get_value(model.trainable_weights)
    saved_optim_weights = K.batch_get_value(model.optimizer.weights)

    test_name = 'test__%f{}.h5'.format(np.random.random())
    modelpath = os.path.join(tempfile.gettempdir(), test_name)
    model.save(modelpath)
    del model
    if tf.__version__[0] == '2' and not TF_KERAS:
        tf.compat.v1.experimental.output_all_intermediates(True)  # bug fix

    model = load_model(modelpath, custom_objects={optimizer_name: optimizer})
    loaded_model_preds = model.predict(X[0])
    loaded_model_weights = K.batch_get_value(model.trainable_weights)
    loaded_optim_weights = K.batch_get_value(model.optimizer.weights)

    assert np.allclose(saved_model_preds, loaded_model_preds,
                       rtol=0, atol=1e-8)
    for smw, lmw in zip(saved_model_weights, loaded_model_weights):
        assert np.allclose(smw, lmw, rtol=0, atol=1e-8)
    for sow, low in zip(saved_optim_weights, loaded_optim_weights):
        assert np.allclose(sow, low, rtol=0, atol=1e-8)


def _make_data(num_batches, batch_size, timesteps, num_channels=None,
               embed_input_dim=None, sparse=False):
    if sparse:
        X = np.random.randint(0, embed_input_dim,
                              (num_batches, batch_size, timesteps))
    else:
        X = np.random.randn(num_batches, batch_size, timesteps, num_channels)
    Y = np.random.randint(0, 2, (num_batches, batch_size))
    return X, Y


def _make_model(batch_shape, l1_reg=None, l2_reg=None, bidirectional=True,
                dense_constraint=None, embed_input_dim=None, sparse=False):
    def _make_reg(l1_reg, l2_reg):
        if l1_reg is not None and l2_reg is None:
            return l1(l1_reg)
        elif l1_reg is None and l2_reg is not None:
            return l2(l2_reg)
        elif l1_reg is not None and l2_reg is not None:
            return l1_l2(l1_reg, l2_reg)
        else:
            return None
    reg = _make_reg(l1_reg, l2_reg)

    if dense_constraint is not None:
        dense_constraint = maxnorm(dense_constraint)

    ipt = Input(batch_shape=batch_shape)
    if sparse:
        x = Embedding(embed_input_dim, embed_input_dim * 3 + 1,
                      mask_zero=True)(ipt)
    else:
        x = ipt
    gru = GRU(4, recurrent_regularizer=reg, bias_regularizer=reg)
    if bidirectional:
        x = Bidirectional(gru)(x)
    else:
        x = gru(x)
    x = Dense(2, kernel_regularizer=reg,
              kernel_constraint=dense_constraint)(x)
    if sparse:
        out = Dense(2, activation='softmax')(x)
    else:
        out = Dense(1, activation='sigmoid')(x)

    return Model(ipt, out)


def _make_optimizer(optimizer_name, model, total_iterations, decay=0,
                    amsgrad=False, nesterov=False, control_mode=False,
                    zero_penalties=True, weight_decays=None):
    optimizers_dict = {'AdamW': AdamW, 'NadamW': NadamW, 'SGDW': SGDW,
                       'Adam': Adam, 'Nadam': Nadam, 'SGD': SGD}
    optimizer = optimizers_dict[optimizer_name]

    optimizer_kw = {}
    if 'Adam' in optimizer_name:
        optimizer_kw = {'amsgrad': amsgrad}
    elif 'SGD' in optimizer_name:
        optimizer_kw = {'nesterov': nesterov, 'momentum': .9}
    if 'Nadam' not in optimizer_name:
        optimizer_kw.update({'decay': decay})

    if not control_mode:
        lr_multipliers = {'gru': 0.5}
        use_cosine_annealing = True
    else:
        lr_multipliers = None
        use_cosine_annealing = False

    if optimizer_name in ('AdamW', 'NadamW', 'SGDW'):
        return optimizer(lr=1e-4, model=model, lr_multipliers=lr_multipliers,
                         use_cosine_annealing=use_cosine_annealing, t_cur=0,
                         total_iterations=total_iterations,
                         weight_decays=weight_decays, **optimizer_kw)
    else:
        return optimizer(lr=1e-4, **optimizer_kw)


def _valid_weight_decays(model):
    weight_decays = get_weight_decays(model)
    return all(x == 0 for l1l2 in weight_decays.values() for x in l1l2)


def _valid_cosine_annealing(eta_history, total_iterations, num_epochs):
    eta_history_simul = []

    for epoch in range(num_epochs):
        for iteration in range(0, total_iterations):
            arg = np.array([np.pi * iteration / (total_iterations - 1)],
                           dtype='float32')
            value = np.array([0.5 * (1 + np.cos(arg))], dtype='float32')
            eta_history_simul.append(value[0][0])
                    # 1 + np.cos(np.pi * iteration / total_iterations)))
    return np.allclose(eta_history, eta_history_simul, rtol=0, atol=2e-7)
