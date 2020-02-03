import os


TF_KERAS = bool(os.environ.get("TF_KERAS", '0') == '1')
TF_2 = bool(os.environ.get("TF_VERSION", '1')[0] == '2')


if TF_KERAS:
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, Embedding
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.constraints import MaxNorm as maxnorm
    from tensorflow.keras.optimizers import Adam, Nadam, SGD
else:
    import keras.backend as K
    from keras.layers import Input, Dense, GRU, Bidirectional, Embedding
    from keras.models import Model, load_model
    from keras.regularizers import l2
    from keras.constraints import MaxNorm as maxnorm
    from keras.optimizers import Adam, Nadam, SGD

if (not TF_2) and TF_KERAS:
    from keras_adamw.utils_225tf import get_weight_decays, fill_dict_in_order
    from keras_adamw.utils_225tf import reset_seeds, K_eval
else:
    from keras_adamw.utils import get_weight_decays, fill_dict_in_order
    from keras_adamw.utils import reset_seeds, K_eval


# ALL TESTS (6 total):
#  - keras    (TF 1.14.0, Keras 2.2.5) [test_optimizers.py]
#  - tf.keras (TF 1.14.0, Keras 2.2.5) [test_optimizers_v2.py]
#  - keras    (TF 2.0.0,  Keras 2.3.0) [test_optimizers.py     --TF_EAGER=True]
#  - keras    (TF 2.0.0,  Keras 2.3.0) [test_optimizers.py     --TF_EAGER=False]
#  - tf.keras (TF 2.0.0,  Keras 2.3.0) [test_optimizers_v2.py, --TF_EAGER=True]
#  - tf.keras (TF 2.0.0,  Keras 2.3.0) [test_optimizers_v2.py, --TF_EAGER=False]
