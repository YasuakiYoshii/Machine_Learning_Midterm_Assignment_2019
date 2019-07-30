import sys, os
sys.path.append(os.pardir) # to import files in parent directory
import numpy as np
import matplotlib.pyplot as plt
import newton_based_method as nb
import batch_steepest_gradient_method as bsg
import multiclass_simple_steepest_gradient_method as mssg
import multiclass_newton_method as mn
from dataset import load4, load5

def compare(method_1 = "bsg", method_2 = "nb", dataset = 4):
    # load dataset
    n = 200
    if dataset == 4:
        x, y = load4(n)
    elif dataset == 5:
        x, y = load5(n)

    # optimize using two methods
    if method_1 == "bsg":
        fun1 = bsg.optimize(x, y)
        label1 = "steepest descent"
    elif method_1 == "nb":
        fun1 = nb.optimize(x, y)
        label1 = "Newton"
    elif method_1 == "mn":
        fun1 = mn.optimize(x, y)
        label1 = "Newton"
    elif method_1 == "mssg":
        fun1 = mssg.optimize(x, y)
        label1 = "simple steepest descent"

    if method_2 == "bsg":
        fun2 = bsg.optimize(x, y)
        label2 = "steepest descent"
    elif method_2 == "nb":
        fun2 = nb.optimize(x, y)
        label2 = "Newton"
    elif method_2 == "mn":
        fun2 = mn.optimize(x, y)
        label2 = "Newton"
    elif method_2 == "mssg":
        fun2 = mssg.optimize(x, y)
        label2 = "simple steepest descent"
    diff_1 = []
    diff_2 = []
    for i in range(len(fun1)-1):
        diff_1.append(fun1[i] - fun1[-1])
    for i in range(len(fun2)-1):
        diff_2.append(fun2[i] - fun2[-1])
    plt.yscale('log')
    plt.plot(np.arange(0, len(diff_1), 1), diff_1, marker="o", color = "red", linewidth=0.5, linestyle = "--", markersize=1, label=label1)
    plt.plot(np.arange(0, len(diff_2), 1), diff_2, marker="v", color = "blue", linewidth=0.5, linestyle = ":", markersize=1, label=label2)
    plt.ylabel("$J(w^{(t)}) - J(\hat{w})$")
    plt.xlabel("iteration")
    #plt.yscale('log')
    plt.xlim(0, max(len(diff_1), len(diff_2)))
    #plt.ylim(0, 0.1)
    plt.legend()
    path = "results/"
    try:
        os.mkdir(path)
    except OSError:
        print ("Creation of the directory %s failed or Already exists" % path)
    else:
        print ("Successfully created the directory %s " % path)
    plt.savefig(path + method_1 + method_2 + ".png")
    plt.savefig(path + method_1 + method_2 + ".eps")
    plt.close()
    return

if __name__ == '__main__':
    compare()

print('module name：{}'.format(__name__))
