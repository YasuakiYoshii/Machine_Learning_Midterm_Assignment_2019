import sys, os
sys.path.append(os.pardir) # to import files in parent directory
import numpy as np
import itertools
import matplotlib.pyplot as plt

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
# 0.9999999
def optimize(rho = 0.99999, eps = 1e-6):

    A = np.array([[250, 15],
              [ 15,  4]])
    mu = np.array([[1],
                   [2]])
    lam = 0.89

    x_init = np.array([[ 3.],
                       [-1.]])
    xt = x_init
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

    step = 0
    u = np.zeros_like(xt)
    v = np.zeros_like(xt)
    x_history = []
    fvalues = []
    g_history = []
    print("Optimizing with AdaDelta ...")
    print("---------------------------------------------------------------")
    print("step".rjust(4) + "  " + "gnorm".ljust(22) + " " + "obj_fun")
    for t in range(100):
        step += 1
        x_history.append(xt.T)
        grad = 2 * np.dot(A, xt-mu)

        g_history.append(grad.flatten().tolist())
        # update w
        v = rho * v + (1. - rho) * grad * grad # v = E[grad^2]
        eta_t = np.sqrt((u + eps) / (v + eps))
        dw = eta_t * grad
        u = rho * u + (1. - rho) * dw * dw # u = E[dw^2]
        xth = xt - dw

        xt = np.array([st_ops(xth[0], lam  * eta_t[0]),
                       st_ops(xth[1], lam  * eta_t[1])])

        fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
        fvalues.append(fv)
        print(str(step).rjust(4) + ": " + str(np.linalg.norm(grad)).ljust(22) + " " + str(np.asscalar(fv)))

    print("---------------------------------------------------------------")
    print("step: " + str(step) + ", gnorm: " + str(np.linalg.norm(grad)) + ", obj_fun: " + str(np.asscalar(fv)))
    print("Omptimization with AdaDelta was done.")
    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)
    return fvalues

def show_loss(history):
    diff = []
    for i in range(len(history)-1):
        diff.append(history[i] - history[-1])
    plt.plot(np.arange(0, len(diff), 1), diff, marker="o", color = "red", linewidth=0.5, linestyle = "--", markersize=1, label="AdaDelta")
    plt.ylabel("diff of func")
    plt.xlabel("iteration")
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    history = optimize()
    show_loss(history)

print('module name：{}'.format(__name__))
