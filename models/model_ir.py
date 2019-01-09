from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theano
import theano.tensor as T
from theano import In, Out

import numpy as np
import timeit
import logging

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)

floatX = theano.config.floatX

def shared_value(data, name, borrow=True):
    return theano.shared(
        np.asarray(data, dtype=floatX),
        name=name,
        borrow=borrow)

def init_weights(shapes, name, borrow=True):
    return theano.shared(
        np.zeros(shapes, dtype=floatX),
        name=name,
        borrow=borrow)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")
    if shuffle:
        numpy.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n//minibatch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start+minibatch_size])
        minibatch_start += minibatch_size
    if(minibatch_start != n):
        minibatches.append(idx_list[minibatch_start:])
    return minibatches

class NegativeBinomialModel(object):
    def __init__(self,
        lo_beta        = 1e-4,
        lo_alpha       = 1e-4,
        hi_alpha       = 3,
        lr             = 1,
        reg_beta       = 0.1,
        reg_alpha      = 0.1,
        decay          = 0.6):

        self.lo_beta   = shared_value(lo_beta, 'lo_beta')
        self.lo_alpha  = shared_value(lo_alpha, 'lo_alpha')
        self.hi_alpha  = shared_value(hi_alpha, 'hi_alpha')
        self.lr        = lr
        self._lr       = shared_value(lr, 'lr')
        self.reg_beta  = shared_value(reg_beta, 'reg_beta')
        self.reg_alpha = shared_value(reg_alpha, 'reg_alpha')
        self.beta      = init_weights(2, 'beta')
        self.alpha     = init_weights(2, 'alpha')
        self.decay     = decay

    def build_model(self):
        # r = mu**2/ (sigma - mu) = 1/alpha
        # p = mu / sigma = 1 / (1+ mu * alpha)
        # mu = beta
        # sigma = mu + alpha*(mu**2)
        y = T.dmatrix('y')
        x = T.imatrix('x')
        self._lambda = T.exp(self.x * self.beta)
        cost = -T.mean(T.gammaln(y + (1 / self.alpha)) - T.gammaln(1 / self.alpha)
               + y * T.log(self.alpha * self._lambda)
               - (y + (1 / self.alpha)) * T.log(1 + self.alpha * self._lambda)) #P81 Cameron (3.26)
        cost += self.reg_beta * T.mean(T.abs_(self.beta)) + self.reg_alpha* T.mean(T.abs_(self.alpha))

        # calculate gradients
        grad_beta    = T.grad(cost, self.beta)
        grad_alpha   = T.grad(cost, self.alpha)
        p_grad_alpha = T.abs_(grad_beta)*grad_alpha + grad_alpha
        p_grad_beta  = T.abs_(grad_alpha)*grad_beta + grad_beta

        # clip and update beta/alpha
        clipped_beta = T.gt(self.beta, self.lo_beta)
        beta_new     = self.beta - self._lr * p_grad_beta * clipped_beta
        beta_new     = T.maximum(self.lo_beta, beta_new)
        alpha_new    = self.alpha - self._lr * p_grad_alpha
        alpha_new    = T.clip(alpha_new, self.lo_alpha, self.hi_alpha)
        updates      = [(self.beta, beta_new), (self.alpha, alpha_new)]

        # compile train function
        start = timeit.default_timer()
        train_func = theano.function(
                inputs=[In(y, borrow=True), In(x, borrow=True)],
                outputs=[self.beta, self.alpha, grad_beta, grad_alpha, cost],
                updates=updates,
                name="train_model"
            )
        end = timeit.default_timer()
        logging.debug("Took %f seconds to build the model" % (end-start))
        return train_func

    def train_model(self,
            data,
            conditions,
            train_func=None,
            n_epochs=1000,
            minibatch_size=None,
            patience = 0):

        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=floatX)

        train_func = train_func or self.build_model()
        n = data.shape[0]
        if (minibatch_size is None) or (minibatch_size > n):
            minibatch_size = n
        minibatches = get_minibatches_idx(n, minibatch_size)

        logger.info("Initializing all variables")
        # initialize beta and alpha by sample means and variances
        lo_alpha    = self.lo_alpha.get_value(borrow=True)
        hi_alpha    = self.hi_alpha.get_value(borrow=True)
        y0          = np.asarray([data[t, :] for t in minibatches[0]], dtype=floatX)
        means       = np.mean(y0, axis=0) + lo_alpha
        variances   = np.var(y0, axis=0) + hi_alpha
        beta_value  = means
        alpha_value = (variances-means)/(means*means)
        alpha_value = np.clip(alpha_value, lo_alpha, hi_alpha)
        self.beta.set_value(beta_value, borrow=True)
        self.alpha.set_value(alpha_value, borrow=True)

        betas        = [beta_value]
        alphas       = [alpha_value]
        grad_betas   = [np.zeros(beta_value.shape)]
        grad_alphas  = [np.zeros(alpha_value.shape)]
        costs        = [0.]

        epoch        = 0
        done_looping = False
        n_batches    = len(minibatches)
        stop_looping = False
        lr           = self.lr

        logger.info("Start training...")
        start = timeit.default_timer()

        X = [[1,0,0,0]] * n # bias, intron, sample, condition
        for i in range(n):
            X[i][2] = 1
            X[i][3] = conditions[i]

        while epoch < n_epochs and not stop_looping:
            lr *= 1./(1.+ self.decay * epoch)
            self._lr.set_value(lr, borrow=True)
            for index in range(n_batches):
                #batch_data = np.asarray([data[t, :] for t in minibatches[index]], dtype=floatX)
                b, a, g_b, g_a, c = train_func(data, X)
                lb = self.lo_beta.get_value(borrow=True)
                betas.append(b)
                alphas.append(a)
                grad_betas.append(g_b)
                grad_alphas.append(g_a)
                costs.append(c)
                if (np.abs(lr*(grad_betas[-1]-grad_betas[-2])).all() < 1e-6
                    and np.abs(lr*(grad_alphas[-1]-grad_alphas[-2])).all() < 1e-6):
                    if patience > 0:
                        patience -= 1
                    else:
                        stop_looping = True
                        break
            epoch = epoch + 1

        end = timeit.default_timer()
        logging.info("MODEL run for %f seconds" % (end-start))
        return betas, alphas, grad_betas, grad_alphas, costs
