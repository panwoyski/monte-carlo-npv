import numpy as np
from tools import async
import matplotlib.pyplot as plt
from scipy.stats import norm


def main():
    mu, sigma = 0, 0.1
    # values = np.random.normal(mu, sigma, size=1000)
    #
    # _, bins, _ = plt.hist(values, 30, normed=True)
    # plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
    #                  np.exp(-(bins - mu)**2 / (2 * sigma**2)),
    #          linewidth=2, color='r')

    # plt.show()

    # parametry sterowane
    # wartosc oczekiwana
    c = mu
    # odchylenie standardowe
    s = sigma
    h = 0.5

    def func(x):
        return norm.cdf(c - x, c, s) - 0.5 * (1 - x * h)

    import scipy.optimize
    a = scipy.optimize.broyden1(func, [1.1], f_tol=1e-14)
    # x = np.linspace(-1,1,100)

    # plt.plot(x, y)
    x = np.linspace(-10, 10, 1000)
    y = async.map(lambda v: scipy.optimize.broyden2(func, [v], f_tol=1e-14), x)
    # y = [scipy.optimize.broyden1(func, [v], f_tol=1e-14) for v in x]
    # y = [func(val) for val in x]
    plt.plot(x, y)
    plt.grid(True)
    plt.show()
    # print(y[0], y[-1])


if __name__ == '__main__':
    main()
