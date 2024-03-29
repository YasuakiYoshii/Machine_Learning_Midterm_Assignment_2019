import sys, os
sys.path.append(os.pardir) # to import files in parent directory
import numpy as np
import itertools
from dataset import load5

def optimize(x, y, lam = 2, n = 200, alpha = 0.02, tol = 1e-6):

    # preprocessing
    x = np.hstack([x, np.ones((n, 1))])
    d = 5
    class_num = 3

    step = 1
    w = np.ones((d,class_num))
    w_history = []
    obj_fun_history = []
    print("Optimizing with Simple Steepest Gradient Method For Multiclass...")
    print("---------------------------------------------------------------")
    print("step".rjust(4) + "  " + "gnorm".ljust(22) + " " + "obj_fun")
    while 1:
        w_history.append(w)
        grad = np.zeros_like(w)
        obj_fun = lam * np.linalg.norm(w)
        L = 0
        for i in range(n):
            yi = y[i].reshape((1,1))
            xi = x[i].reshape((d,1))
            c = np.max(np.dot(w.T, xi)) # avoid overflow
            exps = np.exp(np.dot(w.T, xi) - c)
            softmax = exps / np.sum(exps)
            #print(softmax)
            for j in range(class_num):
                if (j == yi):
                    grad[:,yi] += ((softmax[yi] - 1).reshape((1,1)) * xi).reshape((5,1,1))
                else:
                    grad[:,j] += (softmax[j].reshape((1,1)) * xi).reshape((5,))
            L -= (np.log(softmax[yi]))
            #print(grad)
        grad /= n
        grad += 2 * lam * w
        L /= n
        obj_fun += L
        print(str(step).rjust(4) + ": " + str(np.linalg.norm(grad)).ljust(22) + " " + str(np.asscalar(obj_fun)))
        obj_fun_history.append(np.asscalar(obj_fun))
        #if (np.linalg.norm(grad) < tol):
        if (step >= 200):
            break
        w -= alpha * grad
        step += 1

    print("---------------------------------------------------------------")
    print("step: " + str(step) + ", gnorm: " + str(np.linalg.norm(grad)) + ", obj_fun: " + str(np.asscalar(obj_fun)))
    print("Omptimization was done.")
    return obj_fun_history

if __name__ == '__main__':
    x, y = load5(200)
    optimize(x, y)

print('module name：{}'.format(__name__))
