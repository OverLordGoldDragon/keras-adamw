import os
import tempfile
from unittest import TestCase

import numpy as np
from keras.layers import Input, Dense, GRU, Bidirectional
from keras.models import Model, load_model
from keras.regularizers import l2
import keras.backend as K
import tensorflow as tf
import random
from termcolor import cprint

from keras_adamw.optimizers import AdamW, SGDW, NadamW
from keras_adamw.utils import get_weight_decays, fill_dict_in_order

'''
'''


class TestOptimizers(TestCase):

    def test_all(self):  # Save/Load, Warm Restarts (w/ cosine annealing)
        for optimizer_name, optimizer in zip(['AdamW', 'NadamW', 'SGDW'],
                                             [AdamW, NadamW, SGDW]):
            cprint("<< TESTING {} OPTIMIZER >>".format(optimizer_name), 'blue')
            self._set_random_seed()

            num_batches = 100
            num_epochs = 4
            batch_size, timesteps, input_dim = 32, 10, 8
            batch_shape = (batch_size, timesteps, input_dim)
            total_iterations = num_batches  # due to warm restarts

            self.model = self._make_model(optimizer, batch_shape,
                                          total_iterations)  # for stop-introsp.
            self.assertTrue(self._valid_weight_decays(self.model))

            X, Y = self._make_data(num_batches, *batch_shape)
            self.eta_history = []  # for stop-introspection
            self.t_cur_history = []  # for stop-introspection

            for epoch in range(num_epochs):
                for batch_num in range(num_batches):
                    self.model.train_on_batch(X[batch_num], Y[batch_num])
                    self.t_cur_history += [K.get_value(self.model.optimizer.t_cur)]
                    self.eta_history += [K.get_value(self.model.optimizer.eta_t)]
                K.set_value(self.model.optimizer.t_cur, 0)
            self.assertTrue(self._valid_cosine_annealing(self.eta_history,
                            total_iterations, num_epochs))

            saved_model_preds = self.model.predict(2*X[0])
            saved_model_weights = self.model.layers[1].get_weights()[0]

            test_num = np.random.random()
            test_name = 'test_{}_%f{}.h5'.format(optimizer_name.lower(), test_num)
            modelpath = os.path.join(tempfile.gettempdir(), test_name)
            self.model.save(modelpath)
            del self.model

            self.model = load_model(modelpath,
                                    custom_objects={optimizer_name: optimizer})
            loaded_model_preds = self.model.predict(2*X[0])
            loaded_model_weights = self.model.layers[1].get_weights()[0]

            self.assertTrue(np.allclose(saved_model_preds,   loaded_model_preds,
                                        rtol=0, atol=1e-8))
            self.assertTrue(np.allclose(saved_model_weights, loaded_model_weights,
                                        rtol=0, atol=1e-8))
            # cleanup
            del self.model
            K.clear_session()

            cprint("\n<< {} TEST PASSED >>\n".format(optimizer_name), 'green')
        cprint("\n<< ALL TESTS PASSED >>\n", 'green')

    @staticmethod
    def _make_data(num_batches, batch_size, timesteps, input_dim):
        X = np.random.randn(num_batches, batch_size, timesteps, input_dim)
        Y = np.random.randint(0, 2, (num_batches, batch_size))
        return X, Y

    @staticmethod
    def _make_model(optimizer, batch_shape, total_iterations) -> Model:
        ipt = Input(batch_shape=batch_shape)
        x = Bidirectional(GRU(4, recurrent_regularizer=l2(0),
                              bias_regularizer=l2(0)))(ipt)
        x = Dense(2, kernel_regularizer=l2(0))(x)
        out = Dense(1, activation='sigmoid')(x)
        model = Model(ipt, out)

        wd_dict = get_weight_decays(model)
        wd = fill_dict_in_order(wd_dict, [1e-5, 1e-5, 1e-6])
        lr_m = {'gru': 0.5}

        opt = optimizer(lr=1e-4, weight_decays=wd, lr_multipliers=lr_m,
                        use_cosine_annealing=True, t_cur=0,
                        total_iterations=total_iterations)
        model.compile(opt, 'binary_crossentropy')
        return model

    @staticmethod
    def _valid_weight_decays(model):
        weight_decays = get_weight_decays(model)
        trues = 0
        for wd in weight_decays.values():
            trues += (wd != 0)
        return (trues == 0)

    @staticmethod
    def _valid_cosine_annealing(eta_history, total_iterations, num_epochs):
        eta_history_simul = []

        for epoch in range(num_epochs):
            for iteration in range(1, total_iterations + 1):
                eta_history_simul.append(0.5 * (
                        1 + np.cos(np.pi*iteration / (total_iterations + 1))))
        return np.allclose(eta_history, eta_history_simul, rtol=0, atol=2e-7)

    @staticmethod
    def _set_random_seed():
        np.random.seed(42)
        random.seed(100)
        tf.set_random_seed(999)
        print("USING RANDOM SEED")
