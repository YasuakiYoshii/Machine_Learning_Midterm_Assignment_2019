# created by Tatsuya Hiraoka

# requirement
import sys, os
sys.path.append(os.pardir) # to import files in parent directory
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
import itertools

def st_ops(mu, q):
  x_proj = np.zeros(mu.shape)
  for i in range(len(mu)):
    if mu[i] > q:
      x_proj[i] = mu[i] - q
    else:
      if np.abs(mu[i]) < q:
        x_proj[i] = 0
      else:
        x_proj[i] = mu[i] + q;
  return x_proj

# we need to control this parameter to generate multiple figures
lams = [2, 4, 6]
d = 2; # dimention of w

A = np.array([[  3, 0.5],
              [0.5,   1]])
mu = np.array([[1],
               [2]])

# inital x
x_init = np.array([[ 3],
                   [-1]])
L = 1.01 * np.max(np.linalg.eig(2 * A)[0])
plt.figure()
for lam in lams:
    x_history = []
    xt = x_init
    for t in range(100):
      x_history.append(xt.T)
      grad = 2 * np.dot(A, xt-mu)
      xth = xt - 1/L * grad
      xt = st_ops(xth, lam * 1 / L)

    x_history = np.vstack(x_history)

    diff_x = []
    for i in range(len(x_history)-1):
        diff_x.append(np.linalg.norm(x_history[i] - x_history[-1]))
    if lam == 2:
        plt.plot(range(1, len(x_history)), diff_x, 'ro-', markersize=3, linewidth=0.5, label='lambda=2')
    if lam == 4:
        plt.plot(range(1, len(x_history)), diff_x, 'bo-', markersize=3, linewidth=0.5, label='lambda=4')
    if lam == 6:
        plt.plot(range(1, len(x_history)), diff_x, 'ko-', markersize=3, linewidth=0.5, label='lambda=6')

plt.legend()
plt.yscale('log')
plt.xlabel('iteration')
plt.ylabel('$||w-\hat{w}||$')
path = "results/"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed or Already exists" % path)
else:
    print ("Successfully created the directory %s " % path)
plt.savefig(path + 'diff_of_lam.pdf')
plt.show()
