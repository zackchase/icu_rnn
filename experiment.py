import cPickle
import numpy as np
import os
import re
import sys

from progressbar import ProgressBar
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

from evaluation import compute_micro_evaluations
from lib import floatX
from m2m_rnn import M2M_RNN

def train_linear(X, Y, splits, model_config, results_dir, best_k=10, validation_score='f1',
                threshold_score='f1', threshold_criterion='zack', fn_prefix='', label_idx=None):
    label_idx = np.arange(Y.shape[1]) if label_idx is None else label_idx
    best_perf = None
    best_C = None
    best_model = None
    for C in np.logspace(-3,3, num=20):
        sys.stdout.write('Training Ridge Regression with C={0}...'.format(C))
        sys.stdout.flush()
        model = OneVsRestClassifier(LogisticRegression(C=C))
        try:
            model.fit(X[splits[0]], Y[splits[0]])
        except KeyboardInterrupt:
            sys.stdout.write('training interrupted...')
            break
        except:
            raise

        Yp = model.predict_proba(X[splits[1]])
        perf = compute_micro_evaluations(Y[splits[1]][:,label_idx], Yp[:,label_idx], k=best_k,
                                        threshold_score=threshold_score, criterion=threshold_criterion)
        sys.stdout.write(' {0}={1:.4f}'.format(validation_score, perf[validation_score]))
        sys.stdout.flush()
        if best_perf is None or perf[validation_score] > best_perf[validation_score]:
            best_perf = perf
            best_model = model
            best_C = C
            sys.stdout.write(' *BEST')
        sys.stdout.write('\n')

    model_config['C'] = best_C
    cPickle.dump(best_model, open(os.path.join(results_dir, fn_prefix + '-model.pkl'), 'wb'))

    return best_model, model_config

def train_lstm(X, Y, splits, model_config, results_dir, best_k=10, validation_score='f1',
                threshold_score='f1', threshold_criterion='zack', fn_prefix='', label_idx=None):

    Y = floatX(Y)
    label_idx = np.arange(Y.shape[1]) if label_idx is None else label_idx
    config = { 'nb_hidden': [64, 64], 'nb_epochs': 1000,
                'alpha': 0.0, 'lambda2': 0.0,
                'clip_at': 0.0, 'scale_norm': 0.0,
                'starting_eta': 32.0, 'minimum_eta': 1.0,
                'half_eta_every': 10 }
    config.update(model_config)
    P = config['nb_input'] = X[0].shape[1]
    K = config['nb_output'] = Y.shape[1]
    config['results_dir'] = results_dir

    print 'LSTM Model Configuration\n----------'
    for k in sorted(config):
        print k, ':', config[k]
    print '----------', '\n'

    nb_hidden = config['nb_hidden']
    nb_epochs = config['nb_epochs']
    eta = config['starting_eta']
    min_eta = config['minimum_eta']
    half_every = config['half_eta_every']
    alpha = config['alpha']
    lambda2 = config['lambda2']
    clip_at = config['clip_at']
    scale_norm = config['scale_norm']
    
    model = M2M_RNN(num_input=P, num_hidden=nb_hidden, num_output=K, clip_at=clip_at, scale_norm=scale_norm)
    perf_hist = []
    best_perf = None
    best_epoch = 0
    train_idx = splits[0]
    valid_idx = splits[1]
    try:
        for epoch in range(1, nb_epochs+1):
            if np.mod(epoch, half_every) == 0:
                eta = np.max([eta/2., min_eta])

            np.random.shuffle(train_idx)

            running_total = 0.
            it = 0
            for idx in train_idx:
                it += 1
                cost, last_step_cost = model.train(X[idx], np.tile(Y[idx], (len(X[idx]), 1)), eta, alpha, lambda2)
                cost = float(cost)
                last_step_cost = float(last_step_cost)
                running_total += last_step_cost
                running_avg = running_total / float(it)
                sys.stdout.write('\repoch {5} (eta={6:.2f}): {0:5d}/{1:5d}, cost: {2:.4f}, last: {3:.4f}, avg: {4:.4f}'.format(it, train_idx.shape[0], cost,
                                                                                                             last_step_cost, running_avg,
                                                                                                             epoch, eta))

            ## Save model to file ##
            sys.stdout.write('...saving...')
            sys.stdout.flush()	
            model.save_model_params_dumb(os.path.join(results_dir, fn_prefix + '-model-epoch{0:04d}.pkl.gz'.format(epoch)))
            sys.stdout.write('\n')

            ## Get validation set performance ##
            sys.stdout.write('epoch {0}: avg: {1:.4f}'.format(epoch, running_avg))
            sys.stdout.flush()
            Yp = np.vstack([ model.predict(X[idx]) for idx in valid_idx ])
            perf = compute_micro_evaluations(Y[valid_idx][:,label_idx], Yp[:,label_idx], k=10, threshold_score=threshold_score, criterion=threshold_criterion)
            sys.stdout.write(' valid: {0:.4f} {1:.4f} {2:.4f}'.format(perf['auroc'], perf['auprc'], perf['f1']))
            if best_perf is None or perf[validation_score] > best_perf[validation_score]:
                best_perf = perf
                best_epoch = epoch
                sys.stdout.write(' *BEST')
            perfs = [ perf ]

            ## Get training set performance every 5 epochs ##
            if np.mod(epoch, 5) == 0:
                Yp = np.vstack([ model.predict(X[idx]) for idx in train_idx ])
                perf = compute_micro_evaluations(Y[train_idx][:,label_idx], Yp[:,label_idx], k=10, threshold_score=threshold_score, criterion=threshold_criterion)
                sys.stdout.write(' train: {0:.4f} {1:.4f} {2:.4f}'.format(perf['auroc'], perf['auprc'], perf['f1']))
            else:
                perf = np.zeros(perfs[0].shape) + np.nan
            perfs.append(perf)
            perf_hist.append(np.vstack(perfs))
            sys.stdout.write('\n')
    
    except KeyboardInterrupt:
        print 'training interrupted'
        model.save_model_params_dumb(os.path.join(results_dir, fn_prefix + '-model-epoch{0:04d}.pkl.gz'.format(epoch)))
    except:
        raise

    model.load_model_params_dumb(os.path.join(results_dir, fn_prefix + '-model-epoch{0:04d}.pkl.gz'.format(best_epoch)))
    model.save_model_params_dumb(os.path.join(results_dir, fn_prefix + '-model-best.pkl.gz'))

    perf_hist = np.dstack(perf_hist) if len(perf_hist) > 0 else np.array([])
    np.savez(os.path.join(results_dir, fn_prefix + 'performance-history.npz'), perf_hist=perf_hist, best_epoch=best_epoch)

    return model, model_config, perf_hist

def load_model(model_type, model_config, model_fn):
    sys.stdout.write('Loading saved ' + model_type + ' from file: ' + model_fn + '...')
    sys.stdout.flush()
    if model_type == 'lstm':
        config = { 'nb_hidden': 64, 'nb_epochs': 1000,
                    'eta': 100., 'alpha': 0.0, 'lambda2': 0.000001,
                    'clip_at': 0.0, 'scale_norm': 0.0}
        config.update(model_config)
        model = M2M_RNN(num_input=config['nb_input'], num_hidden=config['nb_hidden'], num_output=config['nb_output'],
                        clip_at=config['clip_at'], scale_norm=config['scale_norm'])
        model.load_model_params_dumb(model_fn)
    else:
        model = cPickle.load(model_fn)
    sys.stdout.write('DONE!\n')

    return model
