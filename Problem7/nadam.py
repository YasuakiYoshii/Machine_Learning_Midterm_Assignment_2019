import sys, os
sys.path.append(os.pardir) # to import files in parent directory
import numpy as np
import itertools
from dataset import load4
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

def optimize(lr = 0.2, mu_param = 0.5, ups = .9999999, eps = 1e-16):

    A = np.array([[250, 15],
              [ 15,  4]])
    mu = np.array([[1],
                   [2]])
    lam = 0.89

    x_init = np.array([[ 3.],
                       [-1.]])
    x_t = x_init
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

    m_t = np.zeros_like(x_t)
    v_t = np.zeros_like(x_t)
    x_history = []
    fvalues = []
    g_history = []
    step = 0
    print("Optimizing with RMS Prop ...")
    print("---------------------------------------------------------------")
    print("step".rjust(4) + "  " + "gnorm".ljust(22) + " " + "obj_fun")
    for t in range(1,101):
        step += 1
        x_history.append(x_t.T)
        grad = 2 * np.dot(A, x_t-mu)

        g_history.append(grad.flatten().tolist())
        # update w
        mu_t = mu_param * (1. - mu_param**(t-1)) / (1. - mu_param**t)
        mu_tp1 = mu_param * (1. - mu_param**t) / (1. - mu_param**(t+1))
        ups_t = ups * (1 - ups**(t-1)) / (1 - ups**t)
        m_t = mu_t * m_t + (1 - mu_t) * grad
        m_bar_t = mu_tp1 * m_t + (1. - mu_t) * grad # Nesterov
        v_t = ups_t * v_t + (1. - ups_t) * grad**2
        v_bar_t = np.sqrt(v_t) + eps
        eta_t = lr / v_bar_t
        s_t = lr * m_bar_t / v_bar_t
        xth = x_t - s_t

        x_t = np.array([st_ops(xth[0], lam  * eta_t[0]),
                       st_ops(xth[1], lam  * eta_t[1])])

        fv = np.dot(np.dot((x_t - mu).T, A), (x_t - mu)) + lam * (np.abs(x_t[0]) + np.abs(x_t[1]))
        fvalues.append(fv)
        print(str(step).rjust(4) + ": " + str(np.linalg.norm(grad)).ljust(22) + " " + str(np.asscalar(fv)))

    print("---------------------------------------------------------------")
    print("step: " + str(step) + ", gnorm: " + str(np.linalg.norm(grad)) + ", obj_fun: " + str(np.asscalar(fv)))
    print("Omptimization with RMS Prop was done.")
    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)
    return fvalues

def show_loss(history):
    diff = []
    for i in range(len(history)-1):
        diff.append(history[i] - history[-1])
    plt.yscale('log')
    plt.plot(np.arange(0, len(diff), 1), diff, marker="o", color = "red", linewidth=0.5, linestyle = "--", markersize=1, label="Nadam")
    plt.ylabel("diff of func")
    plt.xlabel("iteration")
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    history = optimize()
    show_loss(history)

print('module name：{}'.format(__name__))
