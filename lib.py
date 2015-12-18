import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse

def one_hot(string, prepend_eos=False, append_eos=False):
    """
    Take a string and return a one-hot encoding with ASCII
    """

    length = len(string)

    result = np.zeros((length, 257))
    for i in xrange(length):
        result[i,ord(string[i])] = 1.

    special = np.zeros((1, 257))
    special[0,256] = 1.

    if prepend_eos:
        result = np.vstack([special, result])

    if append_eos:
        result = np.vstack([result, special])

    return result


def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)


def random_weights(shape, name=None):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01), name=name)


def zeros(shape, name=""):
    return theano.shared(floatX(np.zeros(shape)), name=name)


def softmax(X, temperature=1.0):
    e_x = T.exp((X - X.max(axis=1).dimshuffle(0, 'x'))/temperature)
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

# def softmax(X, temperature):
#     e_x = T.exp((X-X.max())/temperature)
#     return e_x / e_x.sum()

def sigmoid(X):
    return 1 / (1 + T.exp(-X))


def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
        return X


def rectify(X):
    return T.maximum(X, 0.)


def clip(X, epsilon):

    return T.maximum(T.minimum(X, epsilon), -1*epsilon)

def scale(X, max_norm):
    curr_norm = T.sum(T.abs_(X))
    return ifelse(T.lt(curr_norm, max_norm), X, max_norm * (X/curr_norm))


def SGD (cost, params, eta, lambda2 = 0.0):
    updates = []
    grads = T.grad(cost=cost, wrt=params)

    for p,g in zip(params, grads):
        updates.append([p, p - eta *( g + lambda2*p)])

    return updates


def momentum(cost, params, caches, eta, rho=.1, clip_at=0.0, scale_norm=0.0, lambda2=0.0):
    updates = []
    grads = T.grad(cost=cost, wrt=params)

    for p, c, g in zip(params, caches, grads):
        if clip_at > 0.0:
            grad = clip(g, clip_at)
        else:
            grad = g

        if scale_norm > 0.0:
            grad = scale(g, scale_norm)

        delta = rho * g + (1-rho) * c
        updates.append([c, delta])
        updates.append([p, p - eta * ( delta + lambda2 * p)])

    return updates


def sample_char(probs):
    return one_hot_to_string(np.random.multinomial(1, probs))


def one_hot_to_string(one_hot):
    return chr(one_hot.nonzero()[0][0])


def get_params(layers):
    params = []
    for layer in layers:
        params += layer.get_params()
    return params


def make_caches(params):
    caches = []
    for p in params:
        caches.append(theano.shared(floatX(np.zeros(p.get_value().shape))))

    return caches


def one_step_updates(layers):
    updates = []

    for layer in layers:
        updates += layer.updates()

    return updates
