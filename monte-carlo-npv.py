import numpy as np
# from tools import async
import matplotlib.pyplot as plt


def main():
    mu, sigma = 0, 0.1
    values = np.random.normal(mu, sigma, size=1000)

    _, bins, _ = plt.hist(values, 30, normed=True)
    plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) *
                     np.exp(-(bins - mu)**2 / (2 * sigma**2)),
             linewidth=2, color='r')

    plt.show()

    # parametr sterowany
    c = mu
    # print


if __name__ == '__main__':
    main()
