import sys, os
sys.path.append(os.pardir) # to import files in parent directory
import numpy as np
import itertools
from dataset import load4

def optimize(x, y, lam = 2, n = 200, alpha = 1, tol = 1e-3):

    # preprocessing
    x = np.hstack([x, np.ones((n, 1))])
    d = 5

    step = 1
    w = np.ones((d,1))
    w_history = []
    obj_fun_history = []
    print("Optimizing with Newton Based Method ...")
    print("---------------------------------------------------------------")
    print("step".rjust(4) + "  " + "gnorm".ljust(22) + " " + "obj_fun")
    while 1:
        w_history.append(w)
        hessian = 2 * lam * np.eye(5)
        grad = 2 * lam * w
        obj_fun = lam * np.dot(w.T, w)
        for i in range(n):
            yi = y[i].reshape((1,1))
            xi = x[i].reshape((5,1))
            exp = np.exp(-yi * np.dot(w.T, xi))
            hessian += yi**2 * exp / (1 + exp)**2 * np.dot(xi, xi.T)
            grad -= (yi * xi * exp) / (1 + exp)
            obj_fun += np.log(1 + exp)
        print(str(step).rjust(4) + ": " + str(np.linalg.norm(grad)).ljust(22) + " " + str(np.asscalar(obj_fun)))
        obj_fun_history.append(np.asscalar(obj_fun))
        if (np.linalg.norm(grad) < tol):
            break
        d = np.dot(np.linalg.inv(hessian), grad)
        w -= alpha * d
        step += 1

    print("---------------------------------------------------------------")
    print("step: " + str(step) + ", gnorm: " + str(np.linalg.norm(grad)) + ", obj_fun: " + str(np.asscalar(obj_fun)))
    print("Omptimization was done.")
    return obj_fun_history

if __name__ == '__main__':
    x, y = load4(200)
    optimize(x, y)

print('module name：{}'.format(__name__))
