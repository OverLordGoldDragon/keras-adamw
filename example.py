from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.regularizers import l1, l2, l1_l2
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from keras_adamw import AdamW
from keras_adamw.utils import K_eval

#%%############################################################################
ipt   = Input(shape=(120, 4))
x     = LSTM(60, activation='relu', name='lstm_1',
             kernel_regularizer=l1(1e-4), recurrent_regularizer=l2(2e-4))(ipt)
out   = Dense(1, activation='sigmoid', kernel_regularizer=l1_l2(1e-4, 2e-4))(x)
model = Model(ipt, out)

lr_multipliers = {'lstm_1': 0.5}

optimizer = AdamW(lr=1e-4, model=model, lr_multipliers=lr_multipliers,
                  use_cosine_annealing=True, total_iterations=24)
model.compile(optimizer, loss='binary_crossentropy')

#%%############################################################################
eta_history = []
lr_history = []
for epoch in range(3):
    for iteration in range(24):
        x = np.random.rand(10, 120, 4)  # dummy data
        y = np.random.randint(0, 2, (10, 1))  # dummy labels
        loss = model.train_on_batch(x, y)
        eta_history.append(K_eval(model.optimizer.eta_t, K))
        lr_history.append(K_eval(model.optimizer.lr_t, K))
        print("Iter {} loss: {}".format(iteration + 1, "%.3f" % loss))

        # MANUAL OPTION if autorestart=False is used
        # if iteration == (24 - 2):
        #     K.set_value(model.optimizer.t_cur, -1)  # WARM RESTART
    print("EPOCH {} COMPLETED\n".format(epoch + 1))

# learning rate at iteration `t` (lr_t) is subject to scaling depending on
# optimizer; Adam and Nadam use betas (1 & 2), SGD w/ momentum uses beta.
# -W optimizers additionally scale by eta_t. The lr_t plots reflect the
# ultimate learning rate as a result of all the scalings.

#%%############################################################################
_, ax = plt.subplots(figsize=(10, 5))
ax.plot(eta_history, linewidth=2)
ax.set_xlim(0, len(eta_history))
ax.set_ylim(0, 1.05)
ax.set_ylabel('eta_t', weight='bold', fontsize=15)
ax.set_xlabel('Train iterations', weight='bold', fontsize=15)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))

_, ax = plt.subplots(figsize=(10, 5))
ax.plot(lr_history, linewidth=2)
ax.set_xlim(0, len(lr_history))
ax.set_ylim(0, 1.05 * np.max(lr_history))
ax.set_ylabel('lr_t', weight='bold', fontsize=15)
ax.set_xlabel('Train iterations', weight='bold', fontsize=15)
ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0e'))
plt.show()
