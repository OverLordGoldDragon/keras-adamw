# Keras AdamW

[![Build Status](https://travis-ci.com/OverLordGoldDragon/keras-adamw.svg?token=dGKzzAxzJjaRLzddNsCd&branch=master)](https://travis-ci.com/OverLordGoldDragon/keras-adamw)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/1215c1605ad545cba419ee6e5cc870f5)](https://www.codacy.com?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=OverLordGoldDragon/keras-adamw&amp;utm_campaign=Badge_Grade)
![](https://img.shields.io/badge/keras-tensorflow-blue.svg)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

Keras implementation of **AdamW**, **SGDW**, **NadamW**, and **Warm Restarts**, based on paper [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) - plus **Learning Rate Multipliers**

<img src="https://user-images.githubusercontent.com/16495490/65381086-233f7d00-dcb7-11e9-8c83-d0aec7b3663a.png" width="850">

## Features
 - **Weight decay fix**: decoupling L2 penalty from gradient. _Why use?_
   - Weight decay via L2 penalty yields _worse generalization_, due to decay not working properly
   - Weight decay via L2 penalty leads to a _hyperparameter coupling_ with `lr`, complicating search
 - **Warm restarts (WR)**: cosine annealing learning rate schedule. _Why use?_
   - _Better generalization_ and _faster convergence_ was shown by authors for various data and model sizes
 - **LR multipliers**: _per-layer_ learning rate multipliers. _Why use?_
   - _Pretraining_; if adding new layers to pretrained layers, using a global `lr` is prone to overfitting
   
## Usage

### Weight decay 
`AdamW(.., weight_decays=weight_decays)`<br>
Two methods to set `weight_decays = {<weight matrix name>:<weight decay value>,}`:

```python
# 1. Use keras_adamw.utils.py
Dense(.., kernel_regularizer=l2(0)) # set weight decays in layers as usual, but to ZERO
wd_dict = get_weight_decays(model)
ordered_values = [1e-4, 1e-3, ..] # print(wd_dict) to see returned matrix names, note their order
weight_decays = fill_dict_in_order(wd_dict, ordered_values)
```
```python
# 2. Fill manually
model.layers[1].kernel.name # get name of kernel weight matrix of layer indexed 1
weight_decays.update({'conv1d_0/kernel:0':1e-4}) # example
```

### Warm restarts
`AdamW(.., use_cosine_annealing=True, total_iterations=200)` - refer to _Use guidelines_ below

### LR multipliers
`AdamW(.., lr_multipliers=lr_multipliers)` - to get, `{<layer name>:<multiplier value>,}`:

 1. (a) Name every layer to be modified _(recommended)_, e.g. `Dense(.., name='dense_1')` - OR<br>
 (b) Get every layer name, note which to modify: `[print(idx,layer.name) for idx,layer in enumerate(model.layers)]`
 2. (a) `lr_multipliers = {'conv1d_0':0.1} # target layer by full name` - OR<br>
 (b) `lr_multipliers = {'conv1d':0.1}   # target all layers w/ name substring 'conv1d'`
 
 ## Example 
```python
from keras.layers import Input, Dense, LSTM
from keras.models import Model
from keras.regularizers import l2
from keras_adamw.optimizers import AdamW
from keras_adamw.utils import get_weight_decays, fill_dict_in_order
import numpy as np 

ipt   = Input(shape=(120,4))
x     = LSTM(60, activation='relu',    recurrent_regularizer=l2(0), name='lstm_1')(ipt)
out   = Dense(1, activation='sigmoid', kernel_regularizer   =l2(0), name='output')(x)
model = Model(ipt,out)
```
```python
wd_dict        = get_weight_decays(model)                # {'lstm_1/recurrent:0':0,   'output/kernel:0':0}
weight_decays  = fill_dict_in_order(wd_dict,[4e-4,1e-4]) # {'lstm_1/recurrent:0':4e-4,'output/kernel:0':1e-4}
lr_multipliers = {'lstm_1':0.5}

optimizer = AdamW(lr=1e-4, weight_decays=weight_decays, lr_multipliers=lr_multipliers,
                  use_cosine_annealing=True, total_iterations=24)
model.compile(optimizer, loss='binary_crossentropy')
```
```python
for epoch in range(3):
    for iteration in range(24):
        x = np.random.rand(10,120,4) # dummy data
        y = np.random.randint(0,2,(10,1)) # dummy labels
        loss = model.train_on_batch(x,y)
        print("Iter {} loss: {}".format(iteration+1, "%.3f"%loss))
    print("EPOCH {} COMPLETED".format(epoch+1))
    K.set_value(model.optimizer.t_cur, 0) # WARM RESTART: reset cosine annealing argument
```
<img src="https://user-images.githubusercontent.com/16495490/65729113-2063d400-e08b-11e9-8b6a-3a2ea1c62fdd.png" width="450">

(Full example + plot code: [example.py](https://github.com/OverLordGoldDragon/keras-adamw/blob/master/example.py))

## Use guidelines
### Weight decay
 - **Set L2 penalty to ZERO** if regularizing a weight via `weight_decays` - else the purpose of the 'fix' is largely defeated, and weights will be over-decayed --_My recommendation_
 - `lambda = lambda_norm * sqrt(batch_size/total_iterations)` --> _can be changed_; the intent is to scale λ to _decouple_ it from other hyperparams - including (but _not limited to_), train duration & batch size. --_Authors_ (Appendix, pg.1) (A-1)
 
### Warm restarts
 - Set `t_cur = 0` to restart schedule multiplier (see _Example_). Can be done at compilation or during training. Non-`0` is also valid, and will start `eta_t` at another point on the cosine curve. Details in A-2,3
 - Set `total_iterations` to the # of expected weight updates _for the given restart_ --_Authors_ (A-1,2)
 - `eta_min=0, eta_max=1` are tunable hyperparameters; e.g., an exponential schedule can be used for `eta_max`. If unsure, the defaults were shown to work well in the paper. --_Authors_
 - **[Save/load](https://keras.io/getting-started/faq/#how-can-i-save-a-keras-model) optimizer state**; WR relies on using the optimizer's update history for effective transitions --_Authors_ (A-2)
```python
# 'total_iterations' general purpose example
def get_total_iterations(restart_idx, num_epochs, iterations_per_epoch):
    return num_epochs[restart_idx] * iterations_per_epoch[restart_idx]
get_total_iterations(0, num_epochs=[1,3,5,8], iterations_per_epoch=[240,120,60,30])
```
### Learning rate multipliers
 - Best used for pretrained layers - e.g. greedy layer-wise pretraining, or pretraining a feature extractor to a classifier network. Can be a better alternative to freezing layer weights. --_My recommendation_
 - It's often best not to pretrain layers fully (till convergence, or even best obtainable validation score) - as it may inhibit their ability to adapt to newly-added layers.  --_My recommendation_
 - The more the layers are pretrained, the lower their fraction of new layers' `lr` should be. --_My recommendation_
