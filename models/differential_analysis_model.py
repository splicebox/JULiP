from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import theano
import theano.tensor as T
from theano import In, Out

import numpy as np
import timeit
import logging
from scipy import stats

logging.basicConfig(format="[%(asctime)s] %(message)s", datefmt="%m-%d-%y %H:%M:%S", level=logging.INFO)
logger = logging.getLogger(__name__)


floatX = theano.config.floatX

def shared_value(data, name, borrow=True):
    return theano.shared(
        np.asarray(data, dtype=floatX),
        name=name,
        borrow=borrow)

def init_weights(shapes, name, broadcastable, dtype=floatX, borrow=True):
    if broadcastable == (): # scalar
        return theano.shared(0., name=name,
                            borrow=borrow)
    else:
        return theano.shared(np.zeros(shapes, dtype=dtype),
                            name=name,
                            broadcastable=broadcastable,
                            borrow=borrow)

def get_minibatches_idx(n, minibatch_size, shuffle=False):
    idx_list = np.arange(n, dtype="int32")
    if shuffle:
        np.random.shuffle(idx_list)
    minibatches = []
    minibatch_start = 0
    for i in range(n//minibatch_size):
        minibatches.append(idx_list[minibatch_start: minibatch_start+minibatch_size])
        minibatch_start += minibatch_size
    # if(minibatch_start != n):
    #     minibatches.append(idx_list[minibatch_start:])
    return minibatches

class NegativeBinomialModel(object):
    def __init__(self,
        lo_beta        = 1e-4,
        hi_beta        = 3,
        lo_alpha       = 1e-4,
        hi_alpha       = 1,
        lr             = 0.2,
        # reg_beta       = 0.001,
        # reg_alpha      = 0.001,
        # reg_mu         = 0.001,
        decay          = 0.01,
        alpha_threshold= 0.01):

        self.lo_beta   = shared_value(lo_beta, 'lo_beta')
        self.hi_beta   = shared_value(hi_beta, 'hi_beta')
        self.lo_alpha  = shared_value(lo_alpha, 'lo_alpha')
        self.hi_alpha  = shared_value(hi_alpha, 'hi_alpha')
        self.lr        = lr
        self._lr       = shared_value(lr, 'lr')
        # self.reg_beta  = shared_value(reg_beta, 'reg_beta')
        # self.reg_alpha = shared_value(reg_alpha, 'reg_alpha')
        # self.reg_mu = shared_value(reg_mu, 'reg_mu')
        self.alpha_threshold = alpha_threshold

        self.beta_i    = init_weights((1, 2), 'beta introns', T.row().broadcastable)
        self.beta_ic   = init_weights((1, 2), 'beta control', T.row().broadcastable)
        self.beta_it   = init_weights((1, 2), 'beta test', T.row().broadcastable)

        self.beta_i_   = init_weights((1, 2), 'beta introns for others', T.row().broadcastable)
        #self.beta_ic_  = init_weights((1, 2), 'beta control for others', T.row().broadcastable)
        #self.beta_it_  = init_weights((1, 2), 'beta test for others', T.row().broadcastable)
        self.beta_it_  = init_weights(0, 'beta test for others', T.scalar().broadcastable)

        self.beta_s    = init_weights((2, 1), 'beta samples', T.col().broadcastable)

        self.betas     = [self.beta_i, self.beta_ic, self.beta_it]
        self.betas_    = [self.beta_it_]
        self.betas_shared = [self.beta_s]

        self.alpha     = init_weights((1, 2), 'alpha', T.row().broadcastable)
        self.alpha_    = init_weights((1, 2), 'alpha for others', T.row().broadcastable)
        self.alphas    = [self.alpha, self.alpha_]

        self.decay     = decay
        self.build_model()

    def build_model(self):
        conds = T.dcol('conditions')
        X = T.dmatrix('zero one matrix')

        Y = T.dmatrix('Y')
        mu = (X
            + self.beta_i
            + (conds * self.beta_ic) + ((1 - conds) * self.beta_it)
            + self.beta_s)

        mu = T.exp(mu)
        likelihoods = T.gammaln(Y + (1 / self.alpha)) - T.gammaln(1 / self.alpha) - T.gammaln(Y + 1) \
               + Y * T.log(self.alpha * mu) \
               - (Y + (1 / self.alpha)) * T.log(1 + self.alpha * mu)  # P81 Cameron (3.26)

        Y_ = T.dmatrix('Y for others')
        mu_ = (X
            + self.beta_i
            + (conds * self.beta_ic) + ((1 - conds) * (self.beta_ic + self.beta_it_))
            + self.beta_s)

        mu_ = T.exp(mu_)
        likelihoods_ = T.gammaln(Y_ + (1 / self.alpha_)) - T.gammaln(1 / self.alpha_) - T.gammaln(Y_ + 1) \
               + Y_ * T.log(self.alpha_ * mu_) \
               - (Y_ + (1 / self.alpha_)) * T.log(1 + self.alpha_ * mu_)

        cost1 = -T.mean(likelihoods)  + 0.1 * T.pow(T.sum(T.pow(likelihoods, 2)), 1. / 2)
        cost2 = -T.mean(likelihoods_)  + 0.1 * T.pow(T.sum(T.pow(likelihoods_, 2)), 1. / 2)
        cost = cost1 + cost2
        # cost1 += 5 * T.abs_(T.sum(self.beta_s * conds) / T.sum(conds)
        #                     - T.sum(self.beta_s * (1 - conds)) / T.sum(1 - conds))
        cost1 += T.var(self.beta_s)

        # calculate gradients
        grad_betas = []
        for beta in self.betas:
            grad_beta = T.grad(cost1, beta)
            grad_betas.append(grad_beta)

        grad_betas_ = []
        for beta in self.betas_:
            grad_beta = T.grad(cost2, beta)
            grad_betas_.append(grad_beta)

        grad_betas_shared = []
        for beta in self.betas_shared:
            grad_beta = T.grad(cost1, beta)
            grad_betas_shared.append(grad_beta)

        # clip and update beta/alpha
        updates = []

        grad_alpha = T.grad(cost1, self.alpha)
        grad_beta_i = T.grad(cost1, self.beta_i)
        alpha_mask = T.lt(T.abs_(grad_beta_i), self.alpha_threshold)
        # alpha_new = self.alpha - self._lr * T.clip(grad_alpha, -0.0001, 0.0001)
        alpha_new = self.alpha - self._lr * grad_alpha * alpha_mask
        alpha_new = T.clip(alpha_new, self.lo_alpha, self.hi_alpha)
        updates.append((self.alpha, alpha_new))

        # grad_alpha_ = T.grad(cost, self.alpha_)
        # #grad_beta_i_ = T.grad(cost, self.beta_i_)
        # #alpha_mask_ = T.lt(T.max(T.abs_(grad_beta_i_)), self.alpha_threshold)
        # alpha_new_ = self.alpha_ - self._lr * T.clip(grad_alpha_, -0.0001, 0.0001)
        # alpha_new_ = T.clip(alpha_new_, self.lo_alpha, self.hi_alpha)
        updates.append((self.alpha_, alpha_new))

        grad_alphas = [grad_alpha, grad_alpha]

        for grad_beta, beta in zip(grad_betas + grad_betas_, self.betas + self.betas_):
            beta_new = beta - self._lr * T.clip(grad_beta, -2, 2)
            beta_new = T.clip(beta_new, self.lo_beta, self.hi_beta)
            updates.append((beta, beta_new))

        for grad_beta, beta in zip(grad_betas_shared, self.betas_shared):
            beta_new = beta - self._lr * T.clip(grad_beta, -0.01, 0.01)
            beta_new = T.clip(beta_new, self.lo_beta, self.hi_beta)
            updates.append((beta, beta_new))

        # compile train function
        start = timeit.default_timer()
        outputs = []
        outputs.extend([cost, likelihoods, likelihoods_])
        self.train_func = theano.function(
                inputs=[In(Y, borrow=True), In(Y_, borrow=True), In(X, borrow=True), In(conds, borrow=True)],
                outputs=outputs,
                updates=updates,
                name="train_model"
        )
        end = timeit.default_timer()
        logging.debug("Took %f seconds to build the model" % (end - start))

    def set_beta_value(self, betas, shape, dtype=floatX, base=1e-4):
        beta_value = None
        for beta in betas:
            if beta.type == T.row().type:
                beta_value = np.zeros((1, shape[1]), dtype=dtype) + base
            elif beta.type == T.col().type:
                beta_value = np.zeros((shape[0], 1), dtype=dtype) + base
            elif beta.type == T.scalar().type:
                beta_value = base
            beta.set_value(beta_value, borrow=True)

    def train_model(self,
                    data,
                    data_,
                    conditions,
                    n_epochs=1000):

        if not isinstance(data, np.ndarray):
            data = np.asarray(data, dtype=floatX)

        if not isinstance(data_, np.ndarray):
            data_ = np.asarray(data_, dtype=floatX)

        logger.debug("Initializing all variables ...")
        # initialize beta and alpha by sample means and variances

        alpha_value = np.zeros((1, data.shape[1]), dtype=floatX) + 1e-8
        self.alpha.set_value(alpha_value, borrow=True)
        self.alpha_.set_value(alpha_value, borrow=True)

        self.set_beta_value(self.betas, data.shape)
        self.set_beta_value(self.betas_, data.shape)
        self.set_beta_value(self.betas_shared, data.shape)

        pre_cost = None
        cost = None
        epoch = 0
        # done_looping = False
        stop_looping = False
        lr = self.lr

        logger.debug("Training model ...")
        start = timeit.default_timer()

        X = np.zeros(data.shape, dtype=floatX)

        while epoch < n_epochs and not stop_looping:
            lr *= 1. / (1. + self.decay * epoch)
            self._lr.set_value(lr, borrow=True)
            outputs = self.train_func(data, data_, X, conditions)
            pre_cost = cost
            cost = outputs[0]

            if (pre_cost is not None and (abs(pre_cost - cost) < 1e-4)):
                break

            likelihoods = outputs[1:]

            epoch = epoch + 1
        end = timeit.default_timer()
        logging.debug("Model run for %f seconds" % (end - start))
        return likelihoods

    def predict(self, betas, shape, conditions, condition):
        X = np.zeros(shape, dtype=floatX)
        indices = np.where(conditions == condition)[0]
        conds = conditions[indices]
        conds = conds if condition == 1 else 1 - conds
        beta_i, beta_ic, beta_it, beta_it_, beta_s = betas[-1]

        beta_ict = beta_ic if condition == 1 else beta_it
        mu = X[indices, :]
        mu += beta_i + (conds * beta_ict)
        mu += beta_s[indices, :]
        mu = np.exp(mu)

        X_ = np.zeros(shape, dtype=floatX)
        beta_ict_ = beta_ic if condition == 1 else (beta_ic + beta_it_)
        mu_ = X_[indices, :]
        mu_ += beta_i + (conds * beta_ict_)
        mu_ += beta_s[indices, :]
        mu_ = np.exp(mu_)
        return mu, mu_


    # def different(self, mu0, mu1, threshold=0.02):
    #     p_values = []
    #     for i in range(mu0.shape[1]):
    #         chisq, p = stats.chisquare(f_obs=mu0[:,i], f_exp=mu1[:,i])
    #         p_values.append(p)
    #     if np.any(np.asarray(p_values) < threshold):
    #         return True
    #     else:
    #         return False

