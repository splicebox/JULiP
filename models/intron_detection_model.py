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
        self.build_model()

    def build_model(self):
        # r = mu**2/ (sigma - mu) = 1/alpha
        # p = mu / sigma = 1 / (1+ mu * alpha)
        # mu = beta
        # sigma = mu + alpha*(mu**2)
        y = T.dmatrix('y')
        cost = -T.mean(T.gammaln(y + (1 / self.alpha)) - T.gammaln(1 / self.alpha)
               + y * T.log(self.alpha * self.beta) - T.gammaln(y + 1)
               - (y + (1 / self.alpha)) * T.log(1 + self.alpha * self.beta)) #P81 Cameron (3.26)
        cost += self.reg_beta * T.mean(T.abs_(self.beta)) + self.reg_alpha * T.mean(T.abs_(self.alpha))

        # calculate gradients
        grad_beta    = T.grad(cost, self.beta)
        grad_alpha   = T.grad(cost, self.alpha)
        p_grad_alpha = T.abs_(grad_beta) * grad_alpha + grad_alpha
        p_grad_beta  = T.abs_(grad_alpha) * grad_beta + grad_beta

        # clip and update beta/alpha
        clipped_beta = T.gt(self.beta, self.lo_beta)
        beta_new     = self.beta - self._lr * p_grad_beta * clipped_beta
        beta_new     = T.maximum(self.lo_beta, beta_new)
        alpha_new    = self.alpha - self._lr * p_grad_alpha
        alpha_new    = T.clip(alpha_new, self.lo_alpha, self.hi_alpha)
        updates      = [(self.beta, beta_new), (self.alpha, alpha_new)]

        # compile train function
        start = timeit.default_timer()
        self.train_func = theano.function(
                inputs=[In(y, borrow=True)],
                outputs=[self.beta, cost],
                updates=updates,
                name="train_model"
        )
        end = timeit.default_timer()
        logging.debug("Took %f seconds to build the model" % (end - start))

    def train_model(self,
                    data,
                    n_epochs=1000,
                    minibatch_size=None,
                    patience=0):

        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=floatX)

        logger.debug("Initializing all variables ...")
        # initialize beta and alpha by sample means and variances
        lo_alpha    = self.lo_alpha.get_value(borrow=True)
        hi_alpha    = self.hi_alpha.get_value(borrow=True)
        means       = np.mean(data, axis=0) + lo_alpha
        variances   = np.var(data, axis=0)
        beta_value  = means
        alpha_value = (variances-means)/(means*means)
        alpha_value = np.clip(alpha_value, lo_alpha, hi_alpha)
        self.beta.set_value(beta_value, borrow=True)
        self.alpha.set_value(alpha_value, borrow=True)

        pre_cost = None
        cost = None
        epoch = 0
        stop_looping = False
        lr = self.lr

        logger.debug("Training model...")
        start = timeit.default_timer()
        while epoch < n_epochs and not stop_looping:
            lr *= 1. / (1. + self.decay * epoch)
            self._lr.set_value(lr, borrow=True)
            outs = self.train_func(data)
            pre_cost = cost
            cost = outs[-1]
            if (pre_cost is not None and (abs(pre_cost - cost) < 1e-4)):
                break

            betas = outs[0]

            epoch = epoch + 1

        end = timeit.default_timer()
        logging.debug("MODEL run for %f seconds" % (end - start))
        return betas

