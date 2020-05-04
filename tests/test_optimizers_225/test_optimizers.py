import os
import tempfile
import numpy as np
import tensorflow as tf

from time import time
from termcolor import cprint
from unittest import TestCase

from .. import K
from .. import Input, Dense, GRU, Bidirectional, Embedding
from .. import Model, load_model
from .. import l1, l2, l1_l2
from .. import maxnorm
from .. import Adam, Nadam, SGD
from keras_adamw import AdamW, NadamW, SGDW
from keras_adamw import reset_seeds, get_weight_decays
from keras_adamw import K_eval as KE

K_eval = lambda x: KE(x, K)


print("TF version: %s" % tf.__version__)
TF_EAGER = bool(os.environ["TF_EAGER"] == "True")
if TF_EAGER:
    print("TF running eagerly")
else:
    tf.compat.v1.disable_eager_execution()
    print("TF running in graph mode")


class TestOptimizers(TestCase):

    def test_main(self):  # Save/Load, Warm Restarts (w/ cosine annealing)
        for optimizer_name in ['AdamW', 'NadamW', 'SGDW']:
            cprint("<< TESTING {} OPTIMIZER >>".format(optimizer_name), 'blue')
            reset_seeds()

            num_batches, num_epochs = 25, 4
            batch_size, timesteps, num_channels = 16, 8, 4
            batch_shape = (batch_size, timesteps, num_channels)
            total_iterations = num_batches  # due to warm restarts

            self.model = self._make_model(batch_shape, total_iterations,
                                          l1_reg=1e-4, l2_reg=1e-4)
            optimizer = self._make_optimizer(optimizer_name, self.model,
                                             total_iterations)
            self.model.compile(optimizer, loss='binary_crossentropy')
            self.assertTrue(self._valid_weight_decays(self.model))
            self.model._make_train_function()  # else K.eval before train may fail

            X, Y = self._make_data(num_batches, *batch_shape)
            self.eta_history = []  # for stop-introspection
            self.t_cur_history = []  # for stop-introspection

            for epoch in range(num_epochs):
                for batch_num in range(num_batches):
                    self.t_cur_history += [K_eval(self.model.optimizer.t_cur)]
                    self.eta_history += [K_eval(self.model.optimizer.eta_t)]
                    self.model.train_on_batch(X[batch_num], Y[batch_num])
                K.set_value(self.model.optimizer.t_cur, 0)

            self.assertTrue(self._valid_cosine_annealing(self.eta_history,
                            total_iterations, num_epochs))
            self._test_save_load(self.model, X, optimizer_name, optimizer)

            # cleanup
            del self.model, optimizer
            reset_seeds(reset_graph_with_backend=K)

            cprint("\n<< {} MAIN TEST PASSED >>\n".format(optimizer_name), 'green')
        cprint("\n<< ALL MAIN TESTS PASSED >>\n", 'green')

    def test_misc(self):  # tests of non-main features to improve coverage
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
            total_iterations = 0

            self.model = self._make_model(batch_shape, total_iterations,
                                          embed_input_dim=embed_input_dim,
                                          dense_constraint=1, l2_reg=1e-4,
                                          bidirectional=False, sparse=True)
            optimizer = self._make_optimizer(optimizer_name, self.model,
                                             **optimizer_kw)
            self.model.compile(optimizer, loss='sparse_categorical_crossentropy')
            X, Y = self._make_data(num_batches, *batch_shape,
                                   embed_input_dim=embed_input_dim, sparse=True)

            for batch_num in range(num_batches):
                self.model.train_on_batch(X[batch_num], Y[batch_num])

            self._test_save_load(self.model, X, optimizer_name, optimizer)

            # cleanup
            del self.model, optimizer
            reset_seeds(reset_graph_with_backend=K)

            cprint("\n<< {} MISC TEST PASSED >>\n".format(optimizer_name), 'green')
        cprint("\n<< ALL MISC TESTS PASSED >>\n", 'green')

    def test_control(self):  # tests losses against original optimizers'
        for optimizer_name in ['AdamW', 'NadamW', 'SGDW']:
            cprint("<< TESTING {} OPTIMIZER >>".format(optimizer_name), 'blue')
            pass_txt = "Control Test Passed"

            if optimizer_name == 'AdamW':
                for amsgrad in [True, False]:
                    self._test_control(optimizer_name, amsgrad=amsgrad)
                    print("\n>> AdamW amsgrad={} {}".format(amsgrad, pass_txt))
            elif optimizer_name == 'NadamW':
                self._test_control(optimizer_name)

            elif optimizer_name == 'SGDW':
                for nesterov in [True, False]:
                    self._test_control(optimizer_name, nesterov=nesterov)
                    print("\n>> SGDW nesterov={} {}".format(nesterov, pass_txt))

            o_name = optimizer_name
            cprint("\n<< {} {} >>\n".format(o_name, pass_txt.upper()), 'green')

        cprint("\n<< ALL CONTROL TESTS PASSED >>\n", 'green')

    def _test_control(self, optimizer_name, amsgrad=False, nesterov=False):
        optimizer_kw = dict(total_iterations=0, decay=1e-3,
                            amsgrad=amsgrad, nesterov=nesterov,
                            control_mode=True)
        num_batches = 100
        batch_size, timesteps = 16, 32
        batch_shape = (batch_size, timesteps)
        embed_input_dim = 5
        total_iterations = 0

        model_kw = dict(batch_shape=batch_shape, dense_constraint=1,
                        total_iterations=total_iterations,
                        embed_input_dim=embed_input_dim, l2_reg=0,
                        bidirectional=False, sparse=True)
        loss_name = 'sparse_categorical_crossentropy'
        reset_seeds(verbose=0)
        X, Y = self._make_data(num_batches, *batch_shape,
                               embed_input_dim=embed_input_dim, sparse=True)

        reset_seeds(reset_graph_with_backend=K, verbose=0)
        self.model_custom = self._make_model(**model_kw)
        optimizer_custom = self._make_optimizer(optimizer_name,
                                                self.model_custom,
                                                **optimizer_kw)
        self.model_custom.compile(optimizer_custom, loss=loss_name)
        self.loss_custom = []  # for introspection
        t0 = time()
        for batch_num in range(num_batches):
            self.loss_custom += [self.model_custom.train_on_batch(
                    X[batch_num], Y[batch_num])]
        print("model_custom -- %s batches -- time: %.2f sec" % (num_batches,
                                                                time() - t0))

        reset_seeds(reset_graph_with_backend=K, verbose=0)
        self.model_control = self._make_model(**model_kw)
        optimizer_control = self._make_optimizer(optimizer_name[:-1],
                                                 self.model_control,
                                                 **optimizer_kw)
        self.model_control.compile(optimizer_control, loss=loss_name)
        self.loss_control = []  # for introspection
        t0 = time()
        for batch_num in range(num_batches):
            self.loss_control += [self.model_control.train_on_batch(
                    X[batch_num], Y[batch_num])]
        print("model_control -- %s batches -- time: %.2f sec" % (num_batches,
                                                                 time() - t0))

        loss_diff = np.abs(np.array(self.loss_custom) -
                           np.array(self.loss_control))
        print("%s max loss diff: %e" % (optimizer_name, np.max(loss_diff)))

        self.assertTrue(np.allclose(self.loss_custom, self.loss_control,
                                    rtol=0, atol=1e-3))
        # cleanup
        del self.model_custom, self.model_control
        del optimizer_custom, optimizer_control
        reset_seeds(reset_graph_with_backend=K, verbose=0)

    def _test_save_load(self, model, X, optimizer_name, optimizer):
        saved_model_preds = model.predict(X[0])
        saved_model_weights = K.batch_get_value(model.trainable_weights)
        saved_optim_weights = K.batch_get_value(model.optimizer.weights)

        test_name = 'test__%f{}.h5'.format(np.random.random())
        modelpath = os.path.join(tempfile.gettempdir(), test_name)
        model.save(modelpath)
        del model
        if tf.__version__[0] == '2':
            tf.compat.v1.experimental.output_all_intermediates(True)  # bug fix

        model = load_model(modelpath, custom_objects={optimizer_name: optimizer})
        loaded_model_preds = model.predict(X[0])
        loaded_model_weights = K.batch_get_value(model.trainable_weights)
        loaded_optim_weights = K.batch_get_value(model.optimizer.weights)

        self.assertTrue(np.allclose(saved_model_preds, loaded_model_preds,
                                    rtol=0, atol=1e-8))
        for smw, lmw in zip(saved_model_weights, loaded_model_weights):
            self.assertTrue(np.allclose(smw, lmw, rtol=0, atol=1e-8))
        for sow, low in zip(saved_optim_weights, loaded_optim_weights):
            self.assertTrue(np.allclose(sow, low, rtol=0, atol=1e-8))

    @staticmethod
    def _make_data(num_batches, batch_size, timesteps, num_channels=None,
                   embed_input_dim=None, sparse=False):
        if sparse:
            X = np.random.randint(0, embed_input_dim,
                                  (num_batches, batch_size, timesteps))
        else:
            X = np.random.randn(num_batches, batch_size, timesteps, num_channels)
        Y = np.random.randint(0, 2, (num_batches, batch_size))
        return X, Y

    @staticmethod
    def _make_model(batch_shape, total_iterations, l1_reg=0, l2_reg=0,
                    bidirectional=True, dense_constraint=None,
                    embed_input_dim=None, sparse=False):
        def _make_reg(l1_reg, l2_reg):
            if l1_reg and not l2_reg:
                return l1(l1_reg)
            elif not l1_reg and l2_reg:
                return l2(l2_reg)
            elif l1_reg and l2_reg:
                return l1_l2(l1_reg + l2_reg)
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

    @staticmethod
    def _make_optimizer(optimizer_name, model, total_iterations, decay=0,
                        amsgrad=False, nesterov=False, control_mode=False):
        optimizer_dict = {'AdamW': AdamW, 'NadamW': NadamW, 'SGDW': SGDW,
                          'Adam': Adam, 'Nadam': Nadam, 'SGD': SGD}
        optimizer = optimizer_dict[optimizer_name]

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

        if not any([optimizer_name == name for name in ('Adam', 'Nadam', 'SGD')]):
            return optimizer(model, lr=1e-4, lr_multipliers=lr_multipliers,
                             use_cosine_annealing=use_cosine_annealing, t_cur=0,
                             total_iterations=total_iterations, **optimizer_kw)
        else:
            return optimizer(lr=1e-4, **optimizer_kw)

    @staticmethod
    def _valid_weight_decays(model):
        weight_decays = get_weight_decays(model)
        return all(x == 0 for l1l2 in weight_decays.values() for x in l1l2)

    @staticmethod
    def _valid_cosine_annealing(eta_history, total_iterations, num_epochs):
        eta_history_simul = []

        for epoch in range(num_epochs):
            for iteration in range(0, total_iterations):
                eta_history_simul.append(0.5 * (
                        1 + np.cos(np.pi*iteration / total_iterations)))
        return np.allclose(eta_history, eta_history_simul, rtol=0, atol=2e-7)
