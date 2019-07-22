import sys, os
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

# prepare dataset
np.random.seed(3846)
n = 40
omega = np.random.randn()
noise = 0.8 * np.random.randn(n)

x = np.random.randn(n, 2) + 0
y = 2 * (omega * x[:,0] + x[:,1] + noise > 0) - 1

#plt.plot(np.extract(y>0,x[:,0]),np.extract(y>0,x[:,1]), 'x')
#plt.plot(np.extract(y<0,x[:,0]),np.extract(y<0,x[:,1]), 'o')

# condition
A = np.array([[  3, 0.5],
              [0.5,   1]])
mu = np.array([[1],
               [2]])
w_hat_lam_1 = []
w_hat_lam_2 = []
lams = np.arange(0, 6, 0.1)
# inital x
w_init = np.array([[ 3],
                   [-1]])
L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

for lam in lams:
    ### implement PG ###
    wt = w_init
    for t in range(1000):
        grad = 2 * np.dot(A, wt-mu)
        wth = wt - 1/L * grad
        wt = st_ops(wth, lam * 1 / L)
    w_hat = wt
    w_hat_lam_1.append(w_hat[0])
    w_hat_lam_2.append(w_hat[1])
    ###
plt.plot(lams, w_hat_lam_1, label='w1')
plt.plot(lams, w_hat_lam_2, label='w2')
plt.xlabel('lambda')
plt.ylabel('w')
plt.legend()
path = "results/"
try:
    os.mkdir(path)
except OSError:
    print ("Creation of the directory %s failed or Already exists" % path)
else:
    print ("Successfully created the directory %s " % path)
plt.savefig(path + "lasso_path" + ".png")
plt.savefig(path + "lasso_path" + ".eps")
plt.show()
