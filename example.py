from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.regularizers import l2
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt

from keras_adamw.optimizers import AdamW
from keras_adamw.utils import get_weight_decays, fill_dict_in_order


ipt = Input(shape=(120, 4))
x = LSTM(60, activation='relu', recurrent_regularizer=l2(0), name='lstm_1')(ipt)
out = Dense(1, activation='sigmoid', kernel_regularizer=l2(0), name='output')(x)
model = Model(ipt, out)

wd_dict = get_weight_decays(model)
weight_decays = fill_dict_in_order(wd_dict, [4e-4, 1e-4])
lr_multipliers = {'lstm_1': 0.5}

optimizer = AdamW(lr=1e-4, weight_decays=weight_decays, lr_multipliers=lr_multipliers,
                  use_cosine_annealing=True, total_iterations=24)
model.compile(optimizer, loss='binary_crossentropy')

eta_history = []
for epoch in range(3):
    for iteration in range(24):
        x = np.random.rand(10, 120, 4)  # dummy data
        y = np.random.randint(0, 2, (10, 1))  # dummy labels
        loss = model.train_on_batch(x, y)
        eta_history.append(K.get_value(model.optimizer.eta_t))
        print("Iter {} loss: {}".format(iteration + 1, "%.3f" % loss))
    print("EPOCH {} COMPLETED".format(epoch + 1))
    K.set_value(model.optimizer.t_cur, 0)  # WARM RESTART

plt.plot(eta_history, linewidth=2)
plt.xlim(0, len(eta_history))
plt.ylim(0, 1.05)
plt.ylabel('eta_t', weight='bold', fontsize=15)
plt.xlabel('Train iterations', weight='bold', fontsize=15)
plt.gcf().set_size_inches(10, 5)
plt.show()
