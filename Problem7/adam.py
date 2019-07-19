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

def optimize(b1 = 0.7, b2 = 0.999999999, ee = 1.0e-8, aa= 0.2):
    A = np.array([[250, 15],
              [ 15,  4]])
    mu = np.array([[1],
                   [2]])
    lam = 0.89

    x_init = np.array([[ 3],
                       [-1]])
    xt = x_init
    L = 1.01 * np.max(np.linalg.eig(2 * A)[0])

    x_history = []
    fvalues = []
    g_history = []
    mm = np.zeros((2,1))
    vv = np.zeros((2,1))

    step = 0
    print("Optimizing with Adam ...")
    print("---------------------------------------------------------------")
    print("step".rjust(4) + "  " + "gnorm".ljust(22) + " " + "obj_fun")
    for t in range(1,101):
        step += 1
        x_history.append(xt.T)
        grad = 2 * np.dot(A, xt-mu)

        mm = b1 * mm + (1-b1) * grad
        vv = b2 * vv + (1 - b2) * (grad * grad)

        mmHat = mm / (1-b1**t)
        vvHat = vv / (1-b2**t)

        g_history.append(grad.T)

        rateProx = aa * np.ones((2, 1)) / (np.sqrt(vvHat) + ee)

        xth = xt -  mmHat * rateProx

        xt = np.array([st_ops(xth[0], lam  * rateProx[0]),
                 st_ops(xth[1], lam  * rateProx[1])])

        fv = np.dot(np.dot((xt - mu).T, A), (xt - mu)) + lam * (np.abs(xt[0]) + np.abs(xt[1]))
        fvalues.append(fv)
        print(str(step).rjust(4) + ": " + str(np.linalg.norm(grad)).ljust(22) + " " + str(np.asscalar(fv)))

    print("---------------------------------------------------------------")
    print("step: " + str(step) + ", gnorm: " + str(np.linalg.norm(grad)) + ", obj_fun: " + str(np.asscalar(fv)))
    print("Omptimization with Adam was done.")
    x_history = np.vstack(x_history)
    fvalues = np.vstack(fvalues)
    return fvalues

def show_loss(history):
    diff = []
    for i in range(len(history)-1):
        diff.append(history[i] - history[-1])
    plt.yscale('log')
    plt.plot(np.arange(0, len(diff), 1), diff, marker="o", color = "red", linewidth=0.5, linestyle = "--", markersize=1, label="Adam")
    plt.ylabel("diff of func")
    plt.xlabel("iteration")
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    history = optimize()
    show_loss(history)

print('module name：{}'.format(__name__))
