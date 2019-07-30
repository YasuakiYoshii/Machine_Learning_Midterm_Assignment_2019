# Newton's method for multiclass
# see also https://stats.stackexchange.com/questions/68391/hessian-of-logistic-function

import sys, os
sys.path.append(os.pardir) # to import files in parent directory
import numpy as np
import itertools
from dataset import load5

def optimize(x, y, lam = 2, n = 200, alpha = 1, tol = 1e-6):
    """
    optimizer with Newton's Method

    Parameters
    ----------
    x : ndarray
        input dataset
    y : ndarray
        label dataset
    lam : int or float
        regularization coefficient
    n : int
        size of dataset
    alpha : float
        learning rate
    tol : flaot
        tolerance

    Returns
    ----------
    obj_fun_history : list
        history of loss function
    """

    # preprocessing
    x = np.hstack([x, np.ones((n, 1))])
    d = 5
    class_num = 3

    step = 1
    w = np.ones((d,class_num))
    w_history = []
    obj_fun_history = []
    print("Optimizing with Newton Based Method ...")
    print("---------------------------------------------------------------")
    while 1:
        w_history.append(w)
        hessian = [np.zeros((5,5)), np.zeros((5,5)), np.zeros((5,5))]
        grad = np.zeros_like(w)
        obj_fun = lam * np.linalg.norm(w)
        L = 0
        # batch
        for i in range(n):
            yi = y[i].reshape((1,1))
            xi = x[i].reshape((5,1))
            c = np.max(np.dot(w.T, xi)) # avoid overflow
            exps = np.exp(np.dot(w.T, xi) - c)
            softmax = exps / np.sum(exps)
            for j in range(class_num):
                if (j == yi):
                    grad[:,yi] += ((softmax[yi] - 1).reshape((1,1)) * xi).reshape((5,1,1))
                else:
                    grad[:,j] += (softmax[j].reshape((1,1)) * xi).reshape((5,))
                hessian[j] += (1 - softmax[j]) * softmax[j] * np.dot(xi, xi.T)
            L -= (np.log(softmax[yi]))
        # scaling and regularization
        for j in range(class_num):
            hessian[j] /= n
            hessian[j] += 2 * lam * np.eye(5)
        grad /= n
        grad += 2 * lam * w
        L /= n
        obj_fun += L
        print(str(step).rjust(4) + ": " + str(np.linalg.norm(grad)).ljust(22) + " " + str(np.asscalar(obj_fun)))
        obj_fun_history.append(np.asscalar(obj_fun))
        #if (np.linalg.norm(grad) < tol):
        if (step >= 200):
            break
        # update w
        for j in range(class_num):
            d = np.dot(np.linalg.inv(hessian[j]), grad[:,j])
            w[:,j] -= alpha * d
        step += 1

    print("---------------------------------------------------------------")
    print("step: " + str(step) + ", gnorm: " + str(np.linalg.norm(grad)) + ", obj_fun: " + str(np.asscalar(obj_fun)))
    print("Omptimization was done.")
    return obj_fun_history

if __name__ == '__main__':
    x, y = load5(200)
    optimize(x, y)

print('module name：{}'.format(__name__))
