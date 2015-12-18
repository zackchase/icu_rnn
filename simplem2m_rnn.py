import numpy as np
import theano
import theano.tensor as T
from lstm import InputLayer, SoftmaxLayer, SigmoidLayer, LSTMLayer
from lib import make_caches, get_params, SGD, momentum, one_step_updates


class M2M_RNN:

    def __init__(self, num_input=256, num_hidden=512, num_output=256):
        X = T.matrix()
        Y = T.vector()
        eta = T.scalar()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output

        inputs = InputLayer(X, name="inputs")
        lstm1 = LSTMLayer(num_input, num_hidden, input_layers=[inputs], name="lstm1")
        lstm2 = LSTMLayer(num_hidden, num_hidden, input_layers=[lstm1], name="lstm2")
        sigmoid = SigmoidLayer(num_hidden, num_output, input_layers=[lstm2], name="yhat")

        Y_hat = sigmoid.output()

        self.layers = inputs, lstm1, lstm2, sigmoid

        params = get_params(self.layers)
        caches = make_caches(params)


        #mean_cost = - T.mean( Y * T.log(Y_hat) + (1-Y) * T.log(1-Y_hat) )

        cost = - T.mean( Y * T.log(Y_hat[-1]) + (1-Y) * T.log(1-Y_hat[-1]) )


        updates = momentum(cost, params, caches, eta, clip_at=2.0)

        self.train = theano.function([X, Y, eta], cost, updates=updates, allow_input_downcast=True)

        self.predict=theano.function([X], [Y_hat[-1]], allow_input_downcast=True)


    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()
