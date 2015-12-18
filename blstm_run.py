from multilabel_blstm import ML_BLSTM
from lib import one_hot, one_hot_to_string, floatX
import numpy as np
import theano
import theano.tensor as T
import sys
import random
import evaluation

###############################
#  Prepare the data
###############################


f = open("../icu_old/icu_old-resampled_60min.npz")
g = open("../icu_old/icu_old-labels_all.npz")

a = np.load(f)
b = np.load(g)

X = a["X"]
Y = b["Yd"].astype("float")

f.close()
g.close()

perms = range(len(X))
random.shuffle(perms)

X = X[perms].copy()
Y = Y[perms].copy()

train_size=9000

X_train = X[0:train_size]
Y_train = Y[0:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]

###############################
#  Instantiate the RNN
###############################

rnn = ML_BLSTM(num_input=13, num_hidden=128, num_output=429)


def train(X, Y, iters=100, eta=100, alpha=0.0):
    running_total = 0

    for it in xrange(iters):
        i = random.randint(0, len(X)-1)
        cost, last_step_cost = rnn.train(X[i], np.tile(Y[i], (len(X[i]), 1)), eta, alpha)
        running_total += last_step_cost
        running_avg = running_total / (it + 1.)
        print "iteration: %s, cost: %s, last: %s, avg: %s" % (it, cost, last_step_cost, running_avg)










