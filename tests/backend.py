import os
import tensorflow as tf

#### Environment configs ######################################################
# for testing locally
os.environ['TF_KERAS'] = os.environ.get("TF_KERAS", '1')
os.environ['TF_EAGER'] = os.environ.get("TF_EAGER", '0')
os.environ['USE_GPU']  = os.environ.get("USE_GPU",  '0')


#### Get flags #################################
TF_KERAS = bool(os.environ['TF_KERAS'] == '1')
TF_EAGER = bool(os.environ['TF_EAGER'] == '1')
TF_2 = bool(tf.__version__[0] == '2')

#### GPU/CPU config ############################
if os.environ['USE_GPU'] == '0':
    if TF_2:
        tf.config.set_visible_devices([], 'GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = '-1'

if TF_2:
    USING_GPU = bool(tf.config.list_logical_devices('GPU') != [])
else:
    USING_GPU = bool(tf.config.experimental.list_logical_devices('GPU') != [])

if os.environ['USE_GPU'] == '1' and not USING_GPU:
    raise Exception("requested to use GPU but TF failed to find it")
elif os.environ['USE_GPU'] == '0' and USING_GPU:
    raise Exception("requrested to use CPU, but failed to hide GPU from TF")

#### Graph/Eager config ########################
if not TF_EAGER:
    tf.compat.v1.disable_eager_execution()
elif not TF_2:
    raise Exception("keras-adamw does not support TF1 in Eager execution")

#### Print configs #############################
print(("{}\nTF version: {}\nTF uses {}\nTF executing in {} mode\n"
       "TF_KERAS = {}\n{}\n").format("=" * 80,
                                     tf.__version__,
                                     "GPU"   if USING_GPU else "CPU",
                                     "Eager" if TF_EAGER  else "Graph",
                                     "1"     if TF_KERAS  else "0",
                                     "=" * 80))

#### Imports + Funcs ##########################################################
if TF_KERAS:
    from tensorflow.keras import backend as K
    from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, Embedding
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.regularizers import l1, l2, l1_l2
    from tensorflow.keras.constraints import MaxNorm as maxnorm
    from tensorflow.keras.optimizers import Adam, Nadam, SGD
else:
    from keras import backend as K
    from keras.layers import Input, Dense, GRU, Bidirectional, Embedding
    from keras.models import Model, load_model
    from keras.regularizers import l1, l2, l1_l2
    from keras.constraints import MaxNorm as maxnorm
    from keras.optimizers import Adam, Nadam, SGD

from keras_adamw import get_weight_decays, fill_dict_in_order
from keras_adamw import reset_seeds, K_eval


# ALL TESTS (6 total):
#  - keras    (TF 1.14.0, Keras 2.2.5) [test_optimizers.py]
#  - tf.keras (TF 1.14.0, Keras 2.2.5) [test_optimizers_v2.py]
#  - keras    (TF 2+,     Keras 2.3.0) [test_optimizers.py     --TF_EAGER=1]
#  - keras    (TF 2+,     Keras 2.3.0) [test_optimizers.py     --TF_EAGER=0]
#  - tf.keras (TF 2+,     Keras 2.3.0) [test_optimizers_v2.py, --TF_EAGER=1]
#  - tf.keras (TF 2+,     Keras 2.3.0) [test_optimizers_v2.py, --TF_EAGER=0]
