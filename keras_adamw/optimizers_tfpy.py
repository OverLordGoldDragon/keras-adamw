from tensorflow.python.framework import ops
from tensorflow.keras import backend as K
from tensorflow.python.keras.optimizers import Optimizer
from tensorflow.python.ops import math_ops, state_ops
from .utils import _apply_weight_decays, _compute_eta_t
from .utils import _apply_lr_multiplier, _check_args, K_eval

"""tf.python.keras implementations

!! NOT RECOMMENDED !! -- implementation only for reference
  - runs x10+ slower than OptimizerV2
  - model.save() does not work properly
  - important functionalities may break or change without notice
  - more info: https://stackoverflow.com/questions/58279628/
               what-is-the-difference-between-tf-keras-and-tf-python-keras
"""


class AdamW(Optimizer):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond".

        batch_size:       int >= 1. Train input batch size; used for normalization
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>

        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [2]-Appendix, 2
        t_cur: int. Value to initialize t_cur to - used for 'warm restarts'.
               To be used together with use_cosine_annealing==True
        total_iterations_wd: int / None. If not None, weight_decays will be
                     applied according to total_iterations_wd instead of
                     total_iterations, contrary to authors' scheme. Set to
                     sum(total_iterations) over all restarts to normalize over
                     all epochs. May yield improvement over `None`.
        init_verbose: bool. If True, print weight-name--weight-decay, and
                      lr-multiplier--layer-name value pairs set during
                      optimizer initialization (recommended)

    # <1> - if using 'warm restarts', then refers to total expected iterations
            for a given restart; can be an estimate, and training won't stop
            at iterations == total_iterations. [2]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)
    # References
        - [1][Adam - A Method for Stochastic Optimization]
             (http://arxiv.org/abs/1412.6980v8)
        - [2][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """

    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.,
                 amsgrad=False, batch_size=32, total_iterations=0,
                 total_iterations_wd=None, use_cosine_annealing=False,
                 weight_decays=None, lr_multipliers=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, **kwargs):
        eta_t = kwargs.pop('eta_t', 1.)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.batch_size = K.variable(batch_size, dtype='int64',
                                         name='batch_size')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.initial_decay = decay
        self.amsgrad = amsgrad
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

        self._init_notified = False
        _check_args(total_iterations, use_cosine_annealing, self.weight_decays)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
                1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                      K.dtype(self.decay))))

        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
            t = math_ops.cast(self.iterations, K.floatx())
        self.updates.append(state_ops.assign_add(self.t_cur, 1))

        lr_t = lr * (math_ops.sqrt(1. - math_ops.pow(self.beta_2, t)) /
                     (1. - math_ops.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        total_iterations = self.total_iterations
        # Cosine annealing
        if self.use_cosine_annealing and total_iterations != 0:
            self.eta_t = _compute_eta_t(self)
        self.lr_t = lr_t * self.eta_t  # for external tracking

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # Learning rate multipliers
            if self.lr_multipliers is not None:
                lr_t = _apply_lr_multiplier(self, lr_t, p, force_eager=True)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * math_ops.square(g)
            if self.amsgrad:
                vhat_t = math_ops.maximum(vhat, v_t)
                p_t = p - self.eta_t * lr_t * m_t / (
                        math_ops.sqrt(vhat_t) + self.epsilon)
                self.updates.append(state_ops.assign(vhat, vhat_t))
            else:
                p_t = p - self.eta_t * lr_t * m_t / (
                        math_ops.sqrt(v_t) + self.epsilon)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                p_t = _apply_weight_decays(self, p, p_t, force_eager=True)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'batch_size': int(K.get_value(self.batch_size)),
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'lr_multipliers': self.lr_multipliers,
            'use_cosine_annealing': self.use_cosine_annealing,
            't_cur': int(K.get_value(self.t_cur)),
            'eta_t': int(K_eval(self.eta_t)),
            'eta_min': int(K.get_value(self.eta_min)),
            'eta_max': int(K.get_value(self.eta_max)),
            'init_verbose': self.init_verbose,
            'epsilon': self.epsilon,
            'amsgrad': self.amsgrad
        }
        base_config = super(AdamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class NadamW(Optimizer):
    """Nesterov Adam optimizer.

    Much like Adam is essentially RMSprop with momentum,
    Nadam is Adam RMSprop with Nesterov momentum.

    Default parameters follow those provided in the paper.
    It is recommended to leave the parameters of this optimizer
    at their default values.

    # Arguments
        lr: float >= 0. Learning rate.
        beta_1/beta_2: floats, 0 < beta < 1. Generally close to 1.
        epsilon: float >= 0. Fuzz factor. If `None`, defaults to `K.epsilon()`.

    # Arguments (other): see AdamW

    # References
        - [Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [On the importance of initialization and momentum in deep learning]
          (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
    """

    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 epsilon=None, schedule_decay=0.004,
                 batch_size=32, total_iterations=0,
                 total_iterations_wd=None, use_cosine_annealing=False,
                 weight_decays=None, lr_multipliers=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, **kwargs):
        eta_t = kwargs.pop('eta_t', 1.)
        super(NadamW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.batch_size = K.variable(batch_size, dtype='int64',
                                         name='batch_size')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        if epsilon is None:
            epsilon = K.epsilon()
        self.epsilon = epsilon
        self.schedule_decay = schedule_decay
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

        self._init_notified = False
        _check_args(total_iterations, use_cosine_annealing, self.weight_decays)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = []

        with ops.control_dependencies([state_ops.assign_add(self.iterations, 1)]):
            t = math_ops.cast(self.iterations, K.floatx())
        self.updates.append(state_ops.assign_add(self.t_cur, 1))

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
                math_ops.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
                math_ops.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations, self.m_schedule] + ms + vs

        total_iterations = self.total_iterations
        # Cosine annealing
        if self.use_cosine_annealing and total_iterations != 0:
            self.eta_t = _compute_eta_t(self)
        self.lr_t = self.lr * self.eta_t  # for external tracking

        for p, g, m, v in zip(params, grads, ms, vs):
            # Learning rate multipliers
            lr_t = self.lr
            if self.lr_multipliers is not None:
                lr_t = _apply_lr_multiplier(self, lr_t, p, force_eager=True)

            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * math_ops.square(g)
            v_t_prime = v_t / (1. - math_ops.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

            self.updates.append(state_ops.assign(m, m_t))
            self.updates.append(state_ops.assign(v, v_t))
            p_t = p - self.eta_t*lr_t * m_t_bar / (
                    math_ops.sqrt(v_t_prime) + self.epsilon)

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                p_t = _apply_weight_decays(self, p, p_t, force_eager=True)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'schedule_decay': self.schedule_decay,
            'batch_size': int(K.get_value(self.batch_size)),
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'lr_multipliers': self.lr_multipliers,
            'use_cosine_annealing': self.use_cosine_annealing,
            't_cur': int(K.get_value(self.t_cur)),
            'eta_t': int(K_eval(self.eta_t)),
            'eta_min': int(K.get_value(self.eta_min)),
            'eta_max': int(K.get_value(self.eta_max)),
            'init_verbose': self.init_verbose
        }
        base_config = super(NadamW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class SGDW(Optimizer):
    """Stochastic gradient descent optimizer.

    Includes support for momentum,
    learning rate decay, and Nesterov momentum.

    # Arguments
        lr: float >= 0. Learning rate.
        momentum: float >= 0. Parameter that accelerates SGD
            in the relevant direction and dampens oscillations.
        decay: float >= 0. Learning rate decay over each update.
        nesterov: boolean. Whether to apply Nesterov momentum.

    # Arguments (other): see AdamW
    """

    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False,
                 batch_size=32, total_iterations=0,
                 total_iterations_wd=None, use_cosine_annealing=False,
                 weight_decays=None, lr_multipliers=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, **kwargs):
        eta_t = kwargs.pop('eta_t', 1.)
        super(SGDW, self).__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.batch_size = K.variable(batch_size, dtype='int64',
                                         name='batch_size')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.initial_decay = decay
        self.nesterov = nesterov
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

        self._init_notified = False
        _check_args(total_iterations, use_cosine_annealing, self.weight_decays)

    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [state_ops.assign_add(self.iterations, 1)]
        self.updates.append(state_ops.assign_add(self.t_cur, 1))

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (  # pylint: disable=g-no-augmented-assignment
                1. / (1. + self.decay * math_ops.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        total_iterations = self.total_iterations
        # Cosine annealing
        if self.use_cosine_annealing and total_iterations != 0:
            self.eta_t = _compute_eta_t(self)
        self.lr_t = lr * self.eta_t  # for external tracking

        for p, g, m in zip(params, grads, moments):
            # Learning rate multipliers
            lr_t = lr
            if self.lr_multipliers is not None:
                lr_t = _apply_lr_multiplier(self, lr_t, p, force_eager=True)

            v = self.momentum * m - self.eta_t * lr_t * g  # velocity
            self.updates.append(state_ops.assign(m, v))

            if self.nesterov:
                p_t = p + self.momentum * v - self.eta_t * lr_t * g
            else:
                p_t = p + v

            # Weight decays
            if p.name in self.weight_decays.keys() and total_iterations != 0:
                p_t = _apply_weight_decays(self, p, p_t, force_eager=True)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(state_ops.assign(p, new_p))
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'momentum': float(K.get_value(self.momentum)),
            'decay': float(K.get_value(self.decay)),
            'nesterov': self.nesterov,
            'batch_size': int(K.get_value(self.batch_size)),
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'lr_multipliers': self.lr_multipliers,
            'use_cosine_annealing': self.use_cosine_annealing,
            't_cur': int(K.get_value(self.t_cur)),
            'eta_t': int(K_eval(self.eta_t)),
            'eta_min': int(K.get_value(self.eta_min)),
            'eta_max': int(K.get_value(self.eta_max)),
            'init_verbose': self.init_verbose
        }
        base_config = super(SGDW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
