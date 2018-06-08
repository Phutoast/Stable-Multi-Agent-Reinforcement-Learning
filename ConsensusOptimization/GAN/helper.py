import numpy as np
import scipy.stats as sp
from matplotlib import pyplot as plt 

def kde(mu, tau, bbox=[-5, 5, -5, 5], save_file="", xlabel="", ylabel="", cmap='Blues'):
    values = np.vstack([mu, tau])
    kernel = sp.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1]-bbox[0])/abs(bbox[3]-bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)

    # plt.show()

# Display Result
def display_result(data, cmap='Reds'):
    x_out = np.concatenate([data for i in range(10)], axis=0)
    kde(x_out[:, 0], x_out[:, 1], bbox=[-2, 2, -2, 2], cmap=cmap)
