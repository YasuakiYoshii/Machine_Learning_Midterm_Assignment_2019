import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset import load3
from mpl_toolkits.mplot3d import Axes3D

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

def obj_fun(A, Z, notQ, lam):
    Z_flatten = Z.flatten()
    J = 0
    for i in notQ:
        J += (A[i] - Z_flatten[i])**2
    J += lam * np.linalg.norm(Z, ord='nuc')
    return J

def optimize():
    A, Q = load3()

    lam = 2
    z_init = np.ones_like(A)
    lr = 0.02
    zt = z_init
    obj_fun_history = []
    for t in range(100):
      #x_history.append(xt.T)
      A = A.flatten()
      zt = zt.flatten()
      sub_set = set(list(range(0, len(A)))) - set(Q)
      #print(list(sub_set))
      grad = np.zeros_like(A)
      for i in list(sub_set):
          grad[i] = 2 * (zt[i] - A[i])
      zth = zt.reshape(20, 40) - lr * grad.reshape(20, 40)
      U, s, V = np.linalg.svd(zth)
      zero_mat = np.zeros((20,20))
      S = np.hstack([np.diag(st_ops(s, lam * lr)), zero_mat])
      zt = np.dot(np.dot(U, S), V)


    fig = plt.figure()
    ax = fig.gca(projection='3d')
    X = np.arange(0, 20)
    Y = np.arange(0, 40)
    X, Y = np.meshgrid(X, Y)
    surf = ax.plot_surface(X.T, Y.T, zt, cmap=plt.cm.plasma,
                       linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    fig2, (axL, axR) = plt.subplots(ncols=2, figsize=(10,4))
    plt.figure()
    axL.pcolor(A.reshape(20, 40), cmap=plt.cm.plasma)
    axL.set_title('A')
    axR.pcolor(zt, cmap=plt.cm.plasma)
    axR.set_title('Z')
    #fig2.colorbar()
    fig.show()
    fig2.show()
    fig.savefig('surface_plot.pdf')
    fig2.savefig('color_plots.pdf')

if __name__ == '__main__':
    optimize()

print('module name：{}'.format(__name__))
