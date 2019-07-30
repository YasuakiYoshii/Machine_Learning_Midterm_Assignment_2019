import numpy as np
import cvxpy as cv
import matplotlib.pyplot as plt

def load3(m = 20, n = 40):
    #dataset3
    np.random.seed(1234)
    r = 2
    A = np.dot(np.random.rand(m, r), np.random.rand(r, n)).flatten()
    ninc = 100
    Q = np.random.permutation(m * n)[:ninc]
    A[Q] = None
    A = A.reshape(m, n)
    return A, Q

# Dataset IV
def load4(n = 200):
    print('loading dataset IV .....')
    x = 3 * (np.random.rand(n, 4) - 0.5)
    y = (2 * x[:, 0].reshape((n,1)) - 1 * x[:, 1].reshape((n,1)) + 0.5 + 0.5 * np.random.randn(n, 1)) > 0
    y = 2 * y - 1
    return x, y

# Dataset V
def load5(n = 200):
    print('loading dataset V .....')
    x = 3 * (np.random.rand(n, 4) - 0.5)
    W = np.array([[  2, -1, 0.5],
                  [ -3,  2,   1],
                  [  1,  2,   3]])

    x_with_error = np.dot(np.hstack([x[:, 0:2].reshape((n, 2)), np.ones((n, 1))]), W.T) + 0.5 * np.random.randn(n, 3)
    maxlogit, y = x_with_error.max(axis=1), x_with_error.argmax(axis=1)
    return x, y

# Dataset VI and sample script for group lasso
def load6():
    print('loading dataset VI .....')
    d = 200
    n = 180
    # we consider 5 groups where each group has 40 attributes
    g = []
    for i in range(5):
        g.append(list(range(i*40+1,(i+1)*40+1)))
    x = np.random.randn(n, d)
    noise = 0.5
    # we consider feature in group 1 and group 2 is activated.
    w = np.vstack([20*np.random.randn(80, 1),
                  np.zeros((120,1)),
                  5*np.random.random()])
    x_tilde = np.hstack([x, np.ones((n, 1))])
    y = np.dot(x_tilde, w) + noise * np.random.randn(n, 1)
    lam = 1.0
    A = np.dot(x_tilde.T, x_tilde) + lam * np.eye(d+1)
    B = np.dot(x_tilde.T, y)
    wridge = np.linalg.solve(A, B) # A*wridge = B
    # cvx
    west = cv.Variable((d+1,1))
    obj_fn = 0.5 / n * cv.sum_squares(x_tilde * west - y) \
            + lam * ( cv.norm(west[g[0]], 2) + cv.norm(west[g[1]], 2) \
            + cv.norm(west[g[2]], 2) + cv.norm(west[g[3]], 2) \
            + cv.norm(west[g[4]], 2))
    objective  = cv.Minimize(obj_fn)
    constraints = []
    prob = cv.Problem(objective, constraints)
    result = prob.solve(solver=cv.CVXOPT)
    west = west.value
    #print(west.shape)

    x_test = np.random.randn(n, d)
    x_test_tilde = np.hstack([x_test, np.ones((n, 1))])
    y_test = np.dot(x_test_tilde, w) + noise * np.random.randn(n, 1)
    y_pred = np.dot(x_test_tilde, west)
    #print(np.mean((y_pred - y_test)**2))
    plt.figure() # fig 1
    plt.plot(west[0:d+1], 'r-o', label='group lasso', markersize=1.5, linewidth=0.5)
    plt.plot(w, 'b-*', label='ground truth', markersize=1.5, linewidth=0.5)
    plt.plot(wridge, 'g-+', label='ridge regression', markersize=1.5, linewidth=0.5)
    plt.legend()
    plt.figure() # fig 2
    plt.plot(y_test, y_pred, 'bs', markersize=1)
    plt.xlabel('ground truth')
    plt.ylabel('prediction')
    print("carinality of w hat: %d" % sum(np.abs(west) < 0.01))
    print("carinality of w ground truth: %d" % sum(np.abs(w) < 0.01))
    plt.show()
    return x_tilde, y, w
