from keras import backend as K
from keras.legacy import interfaces
from keras.optimizers import Optimizer
from .utils import _init_weight_decays, _apply_weight_decays, _check_args
from .utils import _apply_lr_multiplier, _update_t_cur_eta_t, _set_autorestart


class AdamW(Optimizer):
    """AdamW optimizer.
    Default parameters follow those provided in the original paper.
    # Arguments
        lr: float >= 0. Learning rate.
        beta_1: float, 0 < beta < 1. Generally close to 1.
        beta_2: float, 0 < beta < 1. Generally close to 1.
        amsgrad: boolean. Whether to apply the AMSGrad variant of this
            algorithm from the paper "On the Convergence of Adam and Beyond".

        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>

        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
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
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False,
                 epsilon=None, decay=0.0, model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, **kwargs):
        if total_iterations > 1:
            weight_decays = _init_weight_decays(model, zero_penalties,
                                                weight_decays)
        eta_t = kwargs.pop('eta_t', 1.)
        super(AdamW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.decay = K.variable(decay, name='decay')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.initial_decay = decay
        self.epsilon = epsilon or K.epsilon()
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.amsgrad = amsgrad
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

        _set_autorestart(self, autorestart, use_cosine_annealing)
        _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = lr  # to print lr_mult setup
        self._init_notified = False

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]

        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        lr_t_premult = lr_t
        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            # Learning rate multipliers
            if self.lr_multipliers is not None:
                lr_t = _apply_lr_multiplier(self, lr_t_premult, p)

            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - self.eta_t * lr_t * m_t / (
                    K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - self.eta_t * lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            # Weight decays
            if p.name in self.weight_decays.keys():
                p_t = _apply_weight_decays(self, p, p_t)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))

        # Cosine annealing
        _update_t_cur_eta_t(self)
        self.lr_t = lr_t * self.eta_t  # for external tracking

        self._init_notified = True
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'decay': float(K.get_value(self.decay)),
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'lr_multipliers': self.lr_multipliers,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            't_cur': int(K.get_value(self.t_cur)),
            'eta_t': float(K.eval(self.eta_t)),
            'eta_min': float(K.get_value(self.eta_min)),
            'eta_max': float(K.get_value(self.eta_max)),
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

        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>

        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [3]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
        eta_min, eta_max: int, int. Min & max values of cosine annealing
                          lr multiplier; [3]-Appendix, 2
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
        at iterations == total_iterations. [3]-Appendix, pg 1
    # <2> - [AdamW Keras Implementation - Github repository]
            (https://github.com/OverLordGoldDragon/keras_adamw)

    # References
        - [1][Nadam report](http://cs229.stanford.edu/proj2015/054_report.pdf)
        - [2][On the importance of initialization and momentum in deep learning]
             (http://www.cs.toronto.edu/~fritz/absps/momentum.pdf)
        - [3][Fixing Weight Decay Regularization in Adam]
             (https://arxiv.org/abs/1711.05101)
    """
    def __init__(self, lr=0.002, beta_1=0.9, beta_2=0.999,
                 schedule_decay=0.004, epsilon=None,
                 model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, **kwargs):
        if total_iterations > 1:
            weight_decays = _init_weight_decays(model, zero_penalties,
                                                weight_decays)
        eta_t = kwargs.pop('eta_t', 1.)
        super(NadamW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.m_schedule = K.variable(1., name='m_schedule')
            self.lr = K.variable(lr, name='lr')
            self.beta_1 = K.variable(beta_1, name='beta_1')
            self.beta_2 = K.variable(beta_2, name='beta_2')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.epsilon = epsilon or K.epsilon()
        self.schedule_decay = schedule_decay
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.use_cosine_annealing = use_cosine_annealing
        self.init_verbose = init_verbose

        _set_autorestart(self, autorestart, use_cosine_annealing)
        _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = lr  # to print lr_mult setup
        self._init_notified = False

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        t = K.cast(self.iterations, K.floatx()) + 1

        # Due to the recommendations in [2], i.e. warming momentum schedule
        momentum_cache_t = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), t * self.schedule_decay)))
        momentum_cache_t_1 = self.beta_1 * (1. - 0.5 * (
            K.pow(K.cast_to_floatx(0.96), (t + 1) * self.schedule_decay)))
        m_schedule_new = self.m_schedule * momentum_cache_t
        m_schedule_next = self.m_schedule * momentum_cache_t * momentum_cache_t_1
        self.updates.append((self.m_schedule, m_schedule_new))

        shapes = [K.int_shape(p) for p in params]
        ms = [K.zeros(shape) for shape in shapes]
        vs = [K.zeros(shape) for shape in shapes]

        self.weights = [self.iterations] + ms + vs

        for p, g, m, v in zip(params, grads, ms, vs):
            # Learning rate multipliers
            lr_t = self.lr
            if self.lr_multipliers is not None:
                lr_t = _apply_lr_multiplier(self, lr_t, p)

            # the following equations given in [1]
            g_prime = g / (1. - m_schedule_new)
            m_t = self.beta_1 * m + (1. - self.beta_1) * g
            m_t_prime = m_t / (1. - m_schedule_next)
            v_t = self.beta_2 * v + (1. - self.beta_2) * K.square(g)
            v_t_prime = v_t / (1. - K.pow(self.beta_2, t))
            m_t_bar = (1. - momentum_cache_t) * g_prime + (
                momentum_cache_t_1 * m_t_prime)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))

            p_t = p - self.eta_t * lr_t * m_t_bar / (
                K.sqrt(v_t_prime) + self.epsilon)

            # Weight decays
            if p.name in self.weight_decays.keys():
                p_t = _apply_weight_decays(self, p, p_t)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))

        # Cosine annealing
        _update_t_cur_eta_t(self)
        self.lr_t = lr_t * self.eta_t  # for external tracking

        self._init_notified = True
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'beta_1': float(K.get_value(self.beta_1)),
            'beta_2': float(K.get_value(self.beta_2)),
            'epsilon': self.epsilon,
            'schedule_decay': self.schedule_decay,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'lr_multipliers': self.lr_multipliers,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            't_cur': int(K.get_value(self.t_cur)),
            'eta_t': float(K.eval(self.eta_t)),
            'eta_min': float(K.get_value(self.eta_min)),
            'eta_max': float(K.get_value(self.eta_max)),
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

        model: keras.Model/tf.keras.Model/None. If not None, automatically
            extracts weight penalties from layers, and overrides `weight_decays`.
        zero_penalties: bool. If True and `model` is passed, will zero weight
            penalties (loss-based). (RECOMMENDED; see README "Use guidelines").
        total_iterations: int >= 0. Total expected iterations / weight updates
                          throughout training, used for normalization; <1>
        lr_multipliers:   dict / None. Name-value pairs specifying per-layer lr
                          multipliers, as {<layer name>:<multiplier value>}; <2>
        weight_decays:    dict / None. Name-value pairs specifying weight decays,
                          as {<weight matrix name>:<weight decay value>}; <2>

        use_cosine_annealing: bool. If True, multiplies lr each train iteration
                              as a function of eta_min, eta_max, total_iterations,
                              and t_cur (current); [2]-Appendix, 2
        autorestart: bool / None. If True, will automatically do Warm Restarts
                     by resetting `t_cur=0` after `total_iterations`. If None,
                     will default to same as `use_cosine_annealing`. If True
                     but `use_cosine_annealing` is False, will raise ValueError.
                     Note: once optimizer is built (happens on first model fit),
                     changing `autorestart` has no effect; optimizer needs to be
                     re-built.
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
    def __init__(self, lr=0.01, momentum=0., nesterov=False, decay=0.0,
                 model=None, zero_penalties=True,
                 total_iterations=0, total_iterations_wd=None,
                 use_cosine_annealing=False, lr_multipliers=None,
                 weight_decays=None, autorestart=None, init_verbose=True,
                 eta_min=0, eta_max=1, t_cur=0, **kwargs):
        if total_iterations > 1:
            weight_decays = _init_weight_decays(model, zero_penalties,
                                                weight_decays)
        eta_t = kwargs.pop('eta_t', 1.)
        super(SGDW, self).__init__(**kwargs)

        with K.name_scope(self.__class__.__name__):
            self.iterations = K.variable(0, dtype='int64', name='iterations')
            self.lr = K.variable(lr, name='lr')
            self.momentum = K.variable(momentum, name='momentum')
            self.decay = K.variable(decay, name='decay')
            self.eta_min = K.constant(eta_min, name='eta_min')
            self.eta_max = K.constant(eta_max, name='eta_max')
            self.eta_t = K.variable(eta_t, dtype='float32', name='eta_t')
            self.t_cur = K.variable(t_cur, dtype='int64', name='t_cur')

        self.initial_decay = decay
        self.total_iterations = total_iterations
        self.total_iterations_wd = total_iterations_wd or total_iterations
        self.nesterov = nesterov
        self.lr_multipliers = lr_multipliers
        self.weight_decays = weight_decays or {}
        self.init_verbose = init_verbose
        self.use_cosine_annealing = use_cosine_annealing

        _set_autorestart(self, autorestart, use_cosine_annealing)
        _check_args(self, total_iterations, use_cosine_annealing, weight_decays)
        self._init_lr = lr  # to print lr_mult setup
        self._init_notified = False

    @interfaces.legacy_get_updates_support
    def get_updates(self, loss, params):
        grads = self.get_gradients(loss, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))
        # momentum
        shapes = [K.int_shape(p) for p in params]
        moments = [K.zeros(shape) for shape in shapes]
        self.weights = [self.iterations] + moments

        for p, g, m in zip(params, grads, moments):
            # Learning rate multipliers
            lr_t = self.lr
            if self.lr_multipliers is not None:
                lr_t = _apply_lr_multiplier(self, lr_t, p)

            v = self.momentum * m - self.eta_t * lr_t * g  # velocity
            self.updates.append(K.update(m, v))

            if self.nesterov:
                p_t = p + self.momentum * v - self.eta_t * lr_t * g
            else:
                p_t = p + v

            # Weight decays
            if p.name in self.weight_decays.keys():
                p_t = _apply_weight_decays(self, p, p_t)
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)
            self.updates.append(K.update(p, new_p))

        # Cosine annealing
        _update_t_cur_eta_t(self)
        self.lr_t = lr_t * self.eta_t  # for external tracking

        self._init_notified = True
        return self.updates

    def get_config(self):
        config = {
            'lr': float(K.get_value(self.lr)),
            'momentum': float(K.get_value(self.momentum)),
            'decay': float(K.get_value(self.decay)),
            'nesterov': self.nesterov,
            'total_iterations': int(self.total_iterations),
            'weight_decays': self.weight_decays,
            'lr_multipliers': self.lr_multipliers,
            'use_cosine_annealing': self.use_cosine_annealing,
            'autorestart': self.autorestart,
            't_cur': int(K.get_value(self.t_cur)),
            'eta_t': float(K.eval(self.eta_t)),
            'eta_min': float(K.get_value(self.eta_min)),
            'eta_max': float(K.get_value(self.eta_max)),
            'init_verbose': self.init_verbose
        }
        base_config = super(SGDW, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
