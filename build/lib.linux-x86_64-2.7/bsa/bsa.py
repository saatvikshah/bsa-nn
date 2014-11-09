from __future__ import division
import numpy as np
from random import random, randint
import math
from joblib import Parallel, delayed


def bsa(model_vars, X, y, up_bound, low_bound):
    thetas = np.zeros([model_vars.theta_count, model_vars.theta_len])
    old_thetas = np.zeros([model_vars.theta_count, model_vars.theta_len])
    fitnessP = np.zeros(model_vars.theta_count)
    fitnessT = np.zeros(model_vars.theta_count)
    for i in range(model_vars.theta_count):
        for j in range(model_vars.theta_len):
            thetas[i, j] = random() * (up_bound - low_bound) + low_bound
            old_thetas[i, j] = random() * (up_bound - low_bound) + low_bound
        fitnessP[i] = model_vars.bsa_cost_fn(thetas[i, :], X, y)
    for iter in range(model_vars.maxiters):
        if model_vars.verbose:
            print "Iter : " + str(iter)
        # Selection-I
        if random() < random():
            old_thetas = thetas
        # np.random.shuffle(old_thetas)
        old_thetas = old_thetas[np.random.randint(model_vars.theta_count, size=model_vars.theta_count), :]
        #Crossover
        map = np.zeros([model_vars.theta_count, model_vars.theta_len])
        if random() < random():
            for i in range(model_vars.theta_count):
                u = np.random.randint(model_vars.theta_len, size=model_vars.theta_len)
                u_factor = math.ceil(model_vars.MIXRATE * random() * model_vars.theta_len)
                map[i, u[0:u_factor]] = 1
        else:
            for i in range(model_vars.theta_count):
                map[i, randint(0, model_vars.theta_len - 1)] = 1
        #Mutation
        T = thetas + map * 3 * random() * (old_thetas - thetas)
        T = model_vars.boundary_control(T, up_bound, low_bound)
        #Selection II
        for i in range(model_vars.theta_count):
            fitnessT[i] = model_vars.bsa_cost_fn(T[i, :], X, y)
            if fitnessT[i] < fitnessP[i]:
                fitnessP[i] = fitnessT[i]
                thetas[i, :] = T[i, :]
        fittest_index = np.argmin(fitnessP)
        globalmin = fitnessP[fittest_index]
        globalminimizer = thetas[fittest_index, :]
        if model_vars.verbose:
            print "Error ==> " + str(globalmin)
    return globalminimizer


class BSA(object):
    def __init__(self, maxiters=200, population_size=100, MIXRATE=1, up_bound=10, lower_bound=-10, regularization=False,
                 verbose=True, lamda=0.1, n_jobs=1):
        self.regularization = regularization
        self.maxiters = maxiters
        self.theta_count = population_size
        self.lamda = lamda
        self.verbose = verbose
        self.MIXRATE = MIXRATE
        self.up_bound = up_bound
        self.low_bound = lower_bound
        self.n_jobs = n_jobs

    def fit(self, X, y):
        self.labels = np.unique(y)
        self.theta_len = X.shape[1]
        if len(self.labels) > 2:
            final_theta = np.array(Parallel(n_jobs=self.n_jobs)(
                delayed(bsa)(self, X,
                             np.array(map(int, y == self.labels[i])),
                             self.up_bound,
                             self.low_bound) for i in range(len(self.labels))
            ))
        elif len(self.labels) == 2:
            final_theta = bsa(self, X, y, self.up_bound, self.low_bound)
        self.soln = final_theta

    def decision_function(self, X):
        htheta = self.sigmoid(np.dot(X, self.soln))
        return htheta

    def predict(self, X):
        if len(self.labels) == 2:
            preds = np.zeros(X.shape[0])
            H = self.sigmoid(np.dot(X, self.soln)).reshape(X.shape[0])
            for i in range(len(H)):
                if (H[i] > 0.5):
                    preds[i] = 1
                else:
                    preds[i] = 0
        elif len(self.labels) > 2:
            H = self.sigmoid(np.dot(X, self.soln.T))
            preds = []
            for index in np.argmax(H, axis=1).tolist():
                preds.append(self.labels[index])
        return preds

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def bsa_cost_fn(self, theta, X, y):
        """Computes Cost Function for Logistic Regression"""
        H = self.sigmoid(np.dot(X, theta))
        if (self.regularization):
            return self.cfs(H.reshape(H.shape[0]), y, "mse") + (self.lamda / 2 * H.shape[0]) * np.sum(
                theta[1:theta.shape[0]] ** 2)
        else:
            return self.cfs(H.reshape(H.shape[0]), y, "mse")

    def cfs(self, arr1, arr2, type):
        """Returns cost function between 2 numpy arrays, 1st : H 2nd : y"""
        if type == "rmse":
            return (np.sum(((arr1 - arr2) ** 2) ** 0.5) / arr2.shape[0])
        elif type == "mse":
            return (np.sum((arr1 - arr2) ** 2) / (arr2.shape[0]))
        elif type == "logcf":
            return (np.sum(-arr2 * np.log(arr1) - (1 - arr2) * np.log(1 - arr1)) / len(arr2))

    def boundary_control(self, t, u_b, d_b):
        theta_count, num_features = t.shape
        for i in range(theta_count):
            for j in range(num_features):
                if t[i, j] > u_b or t[i, j] < d_b:
                    t[i, j] = random() * (u_b - d_b) + d_b
        return t


class BSANN_OVA(object):
    """
    Implements one vs all
    """

    def __init__(self, hidden_layer_size=25, maxiters=200, population_size=100, MIXRATE=1, upper_bound=50,
                 lower_bound=-50, \
                 regularization=False, lamda=0.1, verbose=True, n_jobs=1):
        self.hidden_layer_size = hidden_layer_size
        self.maxiters = maxiters
        self.theta_count = population_size
        self.reg = regularization
        self.lamda = lamda
        self.verbose = verbose
        self.MIXRATE = MIXRATE
        self.up_bound = upper_bound
        self.low_bound = lower_bound
        self.n_jobs = n_jobs

    def fit(self, X, y):
        """
        Implements One vs. All
        :param X:
        :param y:
        :return:
        """
        self.labels = np.unique(y)
        self.input_layer_size = X.shape[1]
        self.theta_len = 1 * (self.hidden_layer_size + 1) + (self.input_layer_size + 1) * self.hidden_layer_size
        if len(self.labels) > 2:  # OVA
            self.num_labels = len(self.labels)
            final_theta = np.array(Parallel(n_jobs=self.n_jobs)(
                delayed(bsa)(self, X,
                             np.array(map(int, y == self.labels[i])),
                             self.up_bound,
                             self.low_bound) for i in range(len(self.labels))
            ))
        elif len(self.labels) == 2:  # Single Class
            self.num_labels = 1
            final_theta = bsa(self, X, y, self.up_bound, self.low_bound)
        self.soln = final_theta

    def predict(self, X):
        preds = []
        if self.num_labels == 1:
            t1, t2 = self.unpack_weights(self.soln)
            htheta = self._forward(X, t1, t2)
            for i in range(len(htheta)):
                if (htheta[i] > 0.5):
                    preds.append(1)
                else:
                    preds.append(0)
        else:
            htheta = np.zeros([self.num_labels, X.shape[0]])
            for i in range(self.num_labels):
                t1, t2 = self.unpack_weights(self.soln[i, :])
                htheta[i, :] = self._forward(X, t1, t2).T
            for index in np.argmax(htheta, axis=0):
                preds.append(self.labels[index])
        return preds

    def bsa_cost_fn(self, t, X, y):
        t1, t2 = self.unpack_weights(t)
        htheta = self._forward(X, t1, t2)
        if self.reg:
            return self.cfs(htheta.reshape(htheta.shape[0]), y, "mse") + (self.lamda / 2 * htheta.shape[0]) * np.sum(
                t[1:t.shape[0]] ** 2)
        else:
            return self.cfs(htheta.reshape(htheta.shape[0]), y, "mse")

    def cfs(self, arr1, arr2, type):
        """Returns cost function between 2 numpy arrays, 1st : H 2nd : y"""
        if type == "rmse":
            return (np.sum(((arr1 - arr2) ** 2) ** 0.5) / arr2.shape[0])
        elif type == "mse":
            return (np.sum((arr1 - arr2) ** 2) / (arr2.shape[0]))
        elif type == "logcf":
            return (np.sum(-arr2 * np.log(arr1) - (1 - arr2) * np.log(1 - arr1)) / len(arr2))

    def _forward(self, X, t1, t2):
        a1 = np.concatenate((np.ones([X.shape[0], 1]), X), axis=1)
        z1 = np.dot(a1, t1.T)
        a2 = self.sigmoid(z1)
        a2 = np.concatenate((np.ones([a2.shape[0], 1]), a2), axis=1)
        z2 = np.dot(a2, t2.T)
        a3 = self.sigmoid(z2)
        H = a3
        return H

    def pack_weights(self, t1, t2):
        return np.concatenate([t1.ravel(), t2.ravel()])

    def unpack_weights(self, t):
        t1 = np.reshape(t[0:(self.hidden_layer_size * (self.input_layer_size + 1))], \
                        [self.hidden_layer_size, self.input_layer_size + 1])
        t2 = np.reshape(t[(self.hidden_layer_size * (self.input_layer_size + 1)):len(t)], \
                        [1, self.hidden_layer_size + 1])
        return t1, t2

    def sigmoid(self, z):
        return (1 / (1 + np.exp(-z)))

    def boundary_control(self, t, u_b, d_b):
        theta_count, num_features = t.shape
        for i in range(theta_count):
            for j in range(num_features):
                if t[i, j] > u_b or t[i, j] < d_b:
                    t[i, j] = random() * (u_b - d_b) + d_b
        return t
