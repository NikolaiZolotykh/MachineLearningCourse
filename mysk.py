import numpy as np
import matplotlib.pyplot as plt

mu1 = (0.4, 0.8)
mu2 = (0.8, 0.6)
mu3 = (0.1, 0.2)
mu4 = (1.0, 0.3)
sigma = 0.3
d = 2

def generate_points(N = 200, seed = 0):
    from sklearn.datasets import make_blobs
    X, y = make_blobs(n_samples = N, centers = [mu1, mu2, mu3, mu4], cluster_std = sigma, 
                      shuffle = False, random_state = seed)
    y[y == 1] = 0
    y[y == 2] = 1
    y[y == 3] = 1
    return X, y

def draw_points(X, y):
    from numpy import array
    from matplotlib.pyplot import scatter
    scatter(X[:, 0], X[:, 1], color = array(('b', 'r'))[y], alpha = 0.5)

def draw_centers(mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4):
    from matplotlib.pyplot import scatter
    scatter([mu1[0], mu2[0]], [mu1[1], mu2[1]], color = 'b', s = 100, alpha = 1)
    scatter([mu3[0], mu4[0]], [mu3[1], mu4[1]], color = 'r', s = 100, alpha = 1)

def normal_density(X, mu, std = sigma):
    return np.exp(-np.sum((X - mu)**2, axis = 1)/(2*std**2))/(2*np.pi*std)

def bayes_predict(X, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4, sigma = sigma):
    return np.argmax(np.vstack([normal_density(X, mu1, sigma) + normal_density(X, mu2, sigma),
                                normal_density(X, mu3, sigma) + normal_density(X, mu4, sigma)]), axis = 0)

def draw_bayes_sep_curve(res = 500, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4, sigma = sigma):
    xx0_min, xx0_max = plt.xlim()
    xx1_min, xx1_max = plt.ylim()
    xx0, xx1 = np.meshgrid(np.linspace(xx0_min, xx0_max, res), np.linspace(xx1_min, xx1_max, res))
    yy = bayes_predict(np.hstack((np.reshape(xx0, (res**2, 1)), np.reshape(xx1, (res**2, 1)))),
                       mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4, sigma = sigma)   
    yy = yy.reshape(xx0.shape)
    xx0_min, xx0_max = plt.xlim()
    xx1_min, xx1_max = plt.ylim()
    plt.contourf(xx0, xx1, yy, 1, alpha = 0.25, colors = ('b', 'r'))
    plt.contour(xx0, xx1, yy, 1, colors = 'k')
    plt.xlim((xx0_min, xx0_max))
    plt.ylim((xx1_min, xx1_max))

def draw_bayes(res = 500, mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4, sigma = sigma):
    xx0_min, xx0_max = plt.xlim()
    xx1_min, xx1_max = plt.ylim()
    xx0, xx1 = np.meshgrid(np.linspace(xx0_min, xx0_max, res), np.linspace(xx1_min, xx1_max, res))
    yy = bayes_predict(np.hstack((np.reshape(xx0, (res**2, 1)), np.reshape(xx1, (res**2, 1)))),
                       mu1 = mu1, mu2 = mu2, mu3 = mu3, mu4 = mu4, sigma = sigma)   
    yy = yy.reshape(xx0.shape)
    xx0_min, xx0_max = plt.xlim()
    xx1_min, xx1_max = plt.ylim()
    plt.contour(xx0, xx1, yy, 1, colors = 'k', linestyles = 'dashed')
    plt.xlim((xx0_min, xx0_max))
    plt.ylim((xx1_min, xx1_max))

def draw_sep_curve(model, res = 500):
    xx0_min, xx0_max = plt.xlim()
    xx1_min, xx1_max = plt.ylim()
    xx0, xx1 = np.meshgrid(np.linspace(xx0_min, xx0_max, res), np.linspace(xx1_min, xx1_max, res))
    yy = model.predict(np.hstack((np.reshape(xx0, (res**2, 1)), np.reshape(xx1, (res**2, 1)))))   
    yy = yy.reshape(xx0.shape)
    plt.contourf(xx0, xx1, yy, 1, alpha = 0.25, colors = ('b', 'r'))
    plt.contour(xx0, xx1, yy, 1, colors = 'k')
    plt.xlim((xx0_min, xx0_max))
    plt.ylim((xx1_min, xx1_max)) 

def abline(a, b, **args):
    x_0, x_1 = plt.xlim()
    y_0, y_1 = plt.ylim()
    plt.plot([x_0, x_1], [a + b*x_0, a + b*x_1], **args)
    plt.xlim((x_0, x_1))
    plt.ylim((y_0, y_1))
    
 