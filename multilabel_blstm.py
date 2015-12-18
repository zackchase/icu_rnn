import numpy as np
import theano
import theano.tensor as T
from lstm import InputLayer, SoftmaxLayer, SigmoidLayer, LSTMLayer, FullyConnectedLayer
from lib import make_caches, get_params, SGD, momentum, one_step_updates, sigmoid


class ML_BLSTM:

    def __init__(self, num_input=256, num_hidden=512, num_output=256):
        X = T.matrix()
        Y = T.matrix()
        eta = T.scalar()
        alpha = T.scalar()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        inputs = InputLayer(X, name="inputs")
        lstm1f = LSTMLayer(num_input, num_hidden, input_layers=[inputs], name="lstm1f")
        lstm1b = LSTMLayer(num_input, num_hidden, input_layers=[inputs], name="lstm1b", go_backwards=True)

        fc = FullyConnectedLayer(2*num_hidden, num_output, input_layers=[lstm1f, lstm1b], name="yhat")

        Y_hat = sigmoid(T.mean(fc.output(), axis=0))

        self.layers = inputs, lstm1f, lstm1b, fc

        params = get_params(self.layers)
        caches = make_caches(params)


        mean_cost = - T.mean( Y * T.log(Y_hat) + (1-Y) * T.log(1-Y_hat) )

        last_step_cost = - T.mean( Y[-1] * T.log(Y_hat[-1]) + (1-Y[-1]) * T.log(1-Y_hat[-1]) )

        cost = alpha * mean_cost + (1-alpha) * last_step_cost

        updates = momentum(cost, params, caches, eta, clip_at=3.0)

        self.train = theano.function([X, Y, eta, alpha], [cost, last_step_cost], updates=updates, allow_input_downcast=True)

        self.predict=theano.function([X], [Y_hat[-1]], allow_input_downcast=True)


    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
