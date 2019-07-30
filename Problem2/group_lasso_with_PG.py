import sys, os
sys.path.append(os.pardir) # to import files in parent directory
import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cv
from dataset import load6

def st_ops(w, q):
    w_proj = np.zeros(w.shape)
    for i in range(len(w)):
      if np.linalg.norm(w[i]) > q:
        w_proj[i] = w[i] - q / np.linalg.norm(w[i]) * w[i]
      else:
        x_proj[i] = 0
    return w_proj

# condition
A = np.array([[  3, 0.5],
              [0.5,   1]])
mu = np.array([[1],
               [2]])
lam = 1.0

x_1 = np.arange(-1.5, 3, 0.01)
x_2 = np.arange(-1.5, 3, 0.02)

X1, X2 = np.mgrid[-1.5:3:0.01, -1.5:3:0.02]
fValue = np.zeros((len(x_1), len(x_2)))
for i in range(len(x_1)):
  for j in range(len(x_2)):
        inr = np.vstack([x_1[i], x_2[j]])
        fValue[i, j] = np.dot(np.dot((inr-mu).T, A), (inr- mu)) + lam * (np.abs(x_1[i]) + np.abs(x_2[j]))

##########################################################

# cvx
west = cv.Variable((2,1))
obj_fn = cv.quad_form(west - mu, A) \
        + lam * ( cv.norm(west[0], 2) + cv.norm(west[1], 2))
objective  = cv.Minimize(obj_fn)
constraints = []
prob = cv.Problem(objective, constraints)
result = prob.solve(solver=cv.CVXOPT)
west = west.value
#print(west.shape)

# proximal gradient method for group lasso
# inital x
w_init = np.array([[ 3],
                   [-1]])
L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

w_history = []
wt = w_init
for t in range(1000):
    w_history.append(wt.T)
    grad = 2 * np.dot(A, wt-mu)
    wth = wt - 1/L * grad
    wt = st_ops(wth, lam * 1 / L)

w_history = np.vstack(w_history)

plt.figure() # fig 3
plt.contour(X1, X2, fValue)
plt.xlabel('w1')
plt.ylabel('w2')
plt.plot(w_history[:,0], w_history[:,1], 'ro-', markersize=3, linewidth=0.5, label='PG with group lasso')
plt.plot(west[0], west[1], 'ko', label='cvx with group lasso')
plt.legend()
plt.savefig("results/group_lasso_path.png")
plt.savefig("results/group_lasso_path.eps")
plt.show()
