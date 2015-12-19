from m2m_rnn import M2M_RNN
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


###############################
#   Edit the path to point to your data
###############################

f = open("../path/to/data")
g = open("../path/to/data")

a = np.load(f)
b = np.load(g)

X = a["X"]
Y = b["Yd"].astype("float")

f.close()
g.close()

############################
#   Load indices to train with a particular shuffle of the data.
#   Useful for reproducibility
############################
# with open("perms.npy") as f:
#     perms = np.load(f)

# X = X[perms].copy()
# Y = Y[perms].copy()


#
#   Set up your train/ test (and validation splits)
#
train_size=N

X_train = X[0:train_size]
Y_train = Y[0:train_size]
X_test = X[train_size:]
Y_test = Y[train_size:]

###############################
#  Instantiate the RNN
###############################

rnn = M2M_RNN(num_input=13, num_hidden=64, num_output=429, clip_at=0.0)


def train(X, Y, iters=100, eta=100, alpha=0.0, lambda2=0.0):
    running_total = 0

    for it in xrange(iters):
        i = random.randint(0, len(X)-1)

        if X[i].shape[0] < 200:
            cost, last_step_cost = rnn.train(X[i], np.tile(Y[i], (len(X[i]), 1)), eta, alpha, lambda2)

        else:
            cost, last_step_cost = rnn.train(X[i][-200:], np.tile(Y[i], (200, 1)), eta, alpha, lambda2)

        running_total += last_step_cost
        running_avg = running_total / (it + 1.)
        print "iteration: %s, cost: %s, last: %s, avg: %s" % (it, cost, last_step_cost, running_avg)










