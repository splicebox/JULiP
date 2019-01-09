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
        lo_beta        = -1,
        hi_beta        = 12,
        lo_alpha       = 1e-4,
        hi_alpha       = 3,
        lr             = 1,
        reg_beta       = 0.2,
        reg_alpha      = 0.1,
        decay          = 0.001):

        self.lo_beta   = shared_value(lo_beta, 'lo_beta')
        self.hi_beta   = shared_value(hi_beta, 'hi_beta')
        self.lo_alpha  = shared_value(lo_alpha, 'lo_alpha')
        self.hi_alpha  = shared_value(hi_alpha, 'hi_alpha')
        self.lr        = lr
        self._lr       = shared_value(lr, 'lr')
        self.reg_beta  = shared_value(reg_beta, 'reg_beta')
        self.reg_alpha = shared_value(reg_alpha, 'reg_alpha')

        self.beta_g    = init_weights(0, 'beta genes', T.scalar().broadcastable)
        self.beta_i    = init_weights((1, 2), 'beta introns', T.row().broadcastable)
        self.beta_ic   = init_weights((1, 2), 'beta introns', T.row().broadcastable)
        self.beta_it   = init_weights((1, 2), 'beta introns', T.row().broadcastable)
        self.beta_s    = init_weights((2, 1), 'beta samples', T.col().broadcastable)
        self.beta_c    = init_weights(0, 'beta control', T.scalar().broadcastable)
        self.beta_t    = init_weights(0, 'beta test', T.scalar().broadcastable)
        self.betas     = [self.beta_g, self.beta_i, self.beta_ic, self.beta_it, self.beta_s, self.beta_c, self.beta_t]

        self.alpha     = init_weights((1,2), 'alpha', T.row().broadcastable)
        self.decay     = decay

    def build_model(self):
        Y = T.dmatrix('Y')
        X = T.dmatrix('zero one matrix')
        conds = T.dcol('conditions')
        #mu = self.beta_g * X + self.beta_i + (conds * self.beta_ic) + ((1 - conds) * self.beta_it) \
        mu = X + self.beta_i + (conds * self.beta_ic) + ((1 - conds) * self.beta_it) \
            + self.beta_s # + (conds * self.beta_c) + ((1 - conds) * self.beta_t)

        self.mu = T.exp(mu)
        cost = -T.mean(T.gammaln(Y + (1 / self.alpha)) - T.gammaln(1 / self.alpha)
               + Y * T.log(self.alpha * self.mu)
               - (Y + (1 / self.alpha)) * T.log(1 + self.alpha * self.mu)) #P81 Cameron (3.26)
        cost += self.reg_alpha* T.mean(T.abs_(self.alpha))
        for beta in self.betas:
            cost += self.reg_beta * T.mean(T.abs_(beta))


        # calculate gradients
        grad_alpha   = T.grad(cost, self.alpha)
        #p_grad_alpha = T.abs_(grad_beta)*grad_alpha + grad_alpha
        grad_betas = []
        for beta in self.betas:
            grad_beta = T.grad(cost, beta)
            #grad_beta  = T.abs_(grad_alpha)*grad_beta + grad_beta
            grad_betas.append(grad_beta)

        # clip and update beta/alpha
        updates = []
        alpha_mask = T.lt(T.max(T.abs_(T.grad(cost, self.beta_i))), 0.3)
        alpha_new = self.alpha - self._lr * grad_alpha*alpha_mask
        alpha_new = T.clip(alpha_new, self.lo_alpha, self.hi_alpha)
        updates.append((self.alpha, alpha_new))
        for grad_beta, beta in zip(grad_betas, self.betas):
            # clipped_beta = T.gt(self.beta, self.lo_beta)
            #beta_new     = self.beta - self._lr * p_grad_beta * clipped_beta
            beta_new     = beta - self._lr * grad_beta
            beta_new     = T.clip(beta_new, self.lo_beta, self.hi_beta)
            updates.append((beta, beta_new))

        # # update reg_beta
        # grad_reg_beta = T.grad(cost, self.reg_beta)
        # reg_beta_mask = T.lt(T.max(T.abs_(T.grad(cost, self.beta_i))), 0.25)
        # reg_beta_new = self.reg_beta - self._lr * grad_reg_beta*reg_beta_mask
        # updates.append((self.reg_beta, reg_beta_new))

        # compile train function
        start = timeit.default_timer()
        outputs = []
        outputs.extend(self.betas)
        outputs.extend(grad_betas)
        outputs.extend([self.alpha, grad_alpha, cost])
        train_func = theano.function(
                inputs=[In(Y, borrow=True), In(X, borrow=True), In(conds, borrow=True)],
                outputs= outputs,
                updates=updates,
                name="train_model"
            )
        end = timeit.default_timer()
        logging.debug("Took %f seconds to build the model" % (end-start))
        return train_func


    def set_beta_value(self, betas, shape, dtype=floatX, epsilon=1e-4):
        beta_value = None
        for beta in betas:
            if beta.type == T.row().type:
                beta_value = np.zeros((1,shape[1]), dtype=dtype) + epsilon
            elif beta.type == T.col().type:
                beta_value = np.zeros((shape[0],1), dtype=dtype) + epsilon
            elif beta.type == T.scalar().type:
                beta_value = epsilon

            beta.set_value(beta_value, borrow=True)

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

        logger.debug("Initializing all variables")
        # initialize beta and alpha by sample means and variances
        lo_alpha    = self.lo_alpha.get_value(borrow=True)
        hi_alpha    = self.hi_alpha.get_value(borrow=True)
        Y0          = np.asarray([data[t, :] for t in minibatches[0]], dtype=floatX)
        means       = np.mean(Y0, axis=0) + lo_alpha
        variances   = np.var(Y0, axis=0) + hi_alpha

        alpha_value = (variances-means)/(means*means)
        alpha_value = np.clip(alpha_value, lo_alpha, hi_alpha)
        alpha_value = np.reshape(alpha_value, (1, alpha_value.shape[0]))
        self.alpha.set_value(alpha_value, borrow=True)

        self.set_beta_value(self.betas, (minibatch_size, data.shape[1]))

        betas        = []
        alphas       = []
        grad_betas   = []
        grad_alphas  = []
        costs        = []

        epoch        = 0
        done_looping = False
        n_batches    = len(minibatches)
        stop_looping = False
        lr           = self.lr

        logger.debug("Start training...")
        start = timeit.default_timer()

        X = np.ones((minibatch_size, data.shape[1]), dtype=floatX)


        while epoch < n_epochs and not stop_looping:
            minibatches = get_minibatches_idx(n, minibatch_size)
            lr *= 1./(1.+ self.decay * epoch)
            self._lr.set_value(lr, borrow=True)
            for index in range(n_batches):
                batch_data = np.asarray([data[i, :] for i in minibatches[index]], dtype=floatX)
                batch_conds = np.asarray([conditions[i] for i in minibatches[index]], dtype=floatX)
                b_g, b_i, b_ic, b_it, b_s, b_c, b_t, \
                g_b_g, g_b_i, g_b_ic, g_b_it, g_b_s, g_b_c, g_b_t, \
                a, g_a, c = train_func(batch_data, X, batch_conds)

                #lb = self.lo_beta.get_value(borrow=True)
                betas.append([b_g, b_i, b_ic, b_it, b_s, b_c, b_t])
                alphas.append(a)
                grad_betas.append([g_b_g, g_b_i, g_b_ic, g_b_it, g_b_s, g_b_c, g_b_t])
                grad_alphas.append(g_a)
                costs.append(c)
                if len(grad_alphas) > 1:
                    grad_flag = True # if difference of all grads < 1e-6
                    if np.all(np.abs(lr*(grad_alphas[-1]-grad_alphas[-2])) >= 1e-6):
                        grad_flag = False
                    for grad_new, grad_old in zip(grad_betas[-1], grad_betas[-2]):
                        if grad_flag and np.all(np.abs(lr*(grad_new - grad_old)) >= 1e-6):
                            grad_flag = False
                            break

                    if grad_flag:
                        if patience > 0:
                            patience -= 1
                        else:
                            stop_looping = True
                            break
            epoch = epoch + 1

        end = timeit.default_timer()
        logging.debug("MODEL run for %f seconds" % (end-start))
        return betas, alphas, grad_betas, grad_alphas, costs

    def predict(self, betas, data, conditions, condition):
        X = np.ones(data.shape, dtype=floatX)
        indices = np.where(conditions==condition)[0]
        conds = conditions[indices]
        beta_g, beta_i, beta_ic, beta_it, beta_s, beta_c, beta_t = betas[-1]
        beta_ct = beta_c if condition==1 else beta_t
        beta_ict = beta_ic if condition==1 else beta_it
        conds = conds if condition==1 else 1-conds
        # mu = beta_g * X[indices, :] + beta_i + (conds * beta_ict) + beta_s[indices, :]+ (conds * beta_ct)
        mu = X[indices, :] + beta_i + (conds * beta_ict) + beta_s[indices, :] #+ (conds * beta_ct)
        mu = np.exp(mu)
        return mu


    def different(self, mu0, mu1, threshold=0.02):
        p_values = []
        for i in range(mu0.shape[1]):
            chisq, p = stats.chisquare(f_obs=mu0[:,i], f_exp=mu1[:,i])
            p_values.append(p)
        if np.any(np.asarray(p_values) < threshold):
            return True
        else:
            return False


