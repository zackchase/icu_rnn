import numpy as np
import theano
import theano.tensor as T
from lstm import InputLayer, SoftmaxLayer, SigmoidLayer, LSTMLayer
from lib import make_caches, get_params, SGD, momentum, one_step_updates, floatX

import gzip
import cPickle


class M2M_RNN:

    def __init__(self, num_input=256, num_hidden=[512,512], num_output=256, clip_at=0.0, scale_norm=0.0):
        X = T.matrix()
        Y = T.matrix()
        eta = T.scalar()
        alpha = T.scalar()
        lambda2 = T.scalar()

        self.num_input = num_input
        self.num_hidden = num_hidden
        self.num_output = num_output
        self.clip_at = clip_at
        self.scale_norm = scale_norm

        inputs = InputLayer(X, name="inputs")
        num_prev = num_input
        prev_layer = inputs
        layers = [ inputs ]
        for i,num_curr in enumerate(num_hidden):
            lstm = LSTMLayer(num_prev, num_curr, input_layers=[prev_layer], name="lstm{0}".format(i+1))
            num_prev = num_curr
            prev_layer = lstm
            layers.append(lstm)
        sigmoid = SigmoidLayer(num_prev, num_output, input_layers=[prev_layer], name="yhat")
        layers.append(sigmoid)
        # lstm1 = LSTMLayer(num_input, num_hidden, input_layers=[inputs], name="lstm1")
        # lstm2 = LSTMLayer(num_hidden, num_hidden, input_layers=[lstm1], name="lstm2")
        # lstm3 = LSTMLayer(num_hidden, num_hidden, input_layers=[lstm2], name="lstm2")
        # sigmoid = SigmoidLayer(num_hidden, num_output, input_layers=[lstm2], name="yhat")

        Y_hat = sigmoid.output()

        #self.layers = inputs, lstm1, lstm2, sigmoid #, lstm3
        self.layers = layers

        params = get_params(self.layers)
        caches = make_caches(params)


        mean_cost = - T.mean( Y * T.log(Y_hat) + (1-Y) * T.log(1-Y_hat) )

        last_step_cost = - T.mean( Y[-1] * T.log(Y_hat[-1]) + (1-Y[-1]) * T.log(1-Y_hat[-1]) )

        cost = alpha * mean_cost + (1-alpha) * last_step_cost

        updates = momentum(cost, params, caches, eta, clip_at=self.clip_at, scale_norm=self.scale_norm, lambda2=lambda2)

        self.train = theano.function([X, Y, eta, alpha, lambda2], [cost, last_step_cost], updates=updates, allow_input_downcast=True)

        self.predict=theano.function([X], [Y_hat[-1]], allow_input_downcast=True)


    def reset_state(self):
        for layer in self.layers:
            layer.reset_state()

    def save_model_params_dumb(self, filename):
        to_save = { 'num_input': self.num_input, 'num_hidden': self.num_hidden, 'num_output': self.num_output,
                    'clip_at': self.clip_at, 'scale_norm': self.scale_norm }
        for l in self.layers:
            for p in l.get_params():
                assert(p.name not in to_save)
                to_save[p.name] = p.get_value()
        with gzip.open(filename, 'wb') as f:
            cPickle.dump(to_save, f)

    def load_model_params_dumb(self, filename):
        f = gzip.open(filename, 'rb')
        to_load = cPickle.load(f)
        assert(to_load['num_input'] == self.num_input)
        assert(to_load['num_output'] == self.num_output)
        
        saved_nb_hidden = to_load['num_hidden']
        try:
            len(saved_nb_hidden)
        except:
            assert(np.all([ saved_nb_hidden == h for h in self.num_hidden ]))
        else:
            assert(len(saved_nb_hidden) == len(self.num_hidden))
            assert(np.all([ h1 == h2 for h1,h2 in zip(saved_nb_hidden, self.num_hidden) ]))
        
        if 'clip_at' in to_load:
            assert(to_load['clip_at'] == self.clip_at)
        if 'scale_norm' in to_load:
            assert(to_load['scale_norm'] == self.scale_norm)

        for l in self.layers:
            for p in l.get_params():
                p.set_value(floatX(to_load[p.name]))
