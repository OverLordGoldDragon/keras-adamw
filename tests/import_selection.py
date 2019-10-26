import os


tf_version = float(os.environ["TF_VERSION"][:3])
tf_keras = bool(os.environ["TF_KERAS"] == "True")
tf_python = bool(os.environ["TF_PYTHON"] == "True")


if tf_version >= 2:
    if tf_keras:
        from keras_adamw.optimizers_v2 import AdamW, NadamW, SGDW
    elif tf_python:
        from keras_adamw.optimizers_tfpy import AdamW, NadamW, SGDW
    else:
        from keras_adamw.optimizers import AdamW, NadamW, SGDW
else:
    if tf_keras:
        from keras_adamw.optimizers_225tf import AdamW, NadamW, SGDW
    else:
        from keras_adamw.optimizers_225 import AdamW, NadamW, SGDW

if tf_keras:
    import tensorflow.keras.backend as K
    from tensorflow.keras.layers import Input, Dense, GRU, Bidirectional, Embedding
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.constraints import MaxNorm as maxnorm
    from tensorflow.keras.optimizers import Adam, Nadam, SGD
elif tf_python:
    import tensorflow.keras.backend as K  # tf.python.keras.backend is very buggy
    from tensorflow.python.keras.layers import Input, Dense, GRU, Bidirectional
    from tensorflow.python.keras.layers import Embedding
    from tensorflow.python.keras.models import Model, load_model
    from tensorflow.python.keras.regularizers import l2
    from tensorflow.python.keras.constraints import MaxNorm as maxnorm
    from tensorflow.python.keras.optimizers import Adam, Nadam, SGD
else:
    import keras.backend as K
    from keras.layers import Input, Dense, GRU, Bidirectional, Embedding
    from keras.models import Model, load_model
    from keras.regularizers import l2
    from keras.constraints import MaxNorm as maxnorm
    from keras.optimizers import Adam, Nadam, SGD

if tf_version >= 2:
    from keras_adamw.utils import get_weight_decays, fill_dict_in_order
    from keras_adamw.utils import reset_seeds, K_eval
else:
    from keras_adamw.utils_225 import get_weight_decays, fill_dict_in_order
    from keras_adamw.utils_225 import reset_seeds, K_eval


# ALL TESTS (7 total):
#  - keras    (TF 1.14.0, Keras 2.2.5) [test_optimizers.py]
#  - tf.keras (TF 1.14.0, Keras 2.2.5) [test_optimizers_v2.py]
#  - keras    (TF 2.0.0,  Keras 2.3.0) [test_optimizers.py     --TF_EAGER=True]
#  - keras    (TF 2.0.0,  Keras 2.3.0) [test_optimizers.py     --TF_EAGER=False]
#  - tf.keras (TF 2.0.0,  Keras 2.3.0) [test_optimizers_v2.py, --TF_EAGER=True]
#  - tf.keras (TF 2.0.0,  Keras 2.3.0) [test_optimizers_v2.py, --TF_EAGER=False]
#  - tf.python.keras (TF 2.0.0,  Keras 2.3.0) [test_optimizers_tfpy.py]
