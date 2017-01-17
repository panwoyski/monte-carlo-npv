import numpy as np
from tools import async
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize


def calculate_single_triangle(c, s, h):
    """
    :param c: wartosc oczekiwana rozkladu normalnego
    :param s: odchylenie standardowe rozkladu normalnego
    :param h: wysokosc trojkata
    :return: krotka (c-a, c, c + a)
    """
    if h <= 0:
        raise ValueError("h(n) musi byc wieksze od zera")
    '''
    Definicja równania którego miejscem zerowym jest wartość a
    norm.cdf - wartosc dystrybuanty w punkcie c-a, dla rozkladu normalnego
               o parametrach zdefiniowanych powyzej
    pozostala czesc rownania generowana ze wzoru
    na pole trojkata oraz ograniczenia (cn - xn)*hn = 1
    '''
    def equation(a):
        return norm.cdf(c - a, c, s) - 0.5 * (1 - a * h)

    '''
    Algorytm Broydena do rozwiazania rownania
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.broyden1.html#scipy.optimize.broyden1
    mozna wybrac dowolny inny solver ukladow nieliniowych
    '''
    a = scipy.optimize.broyden1(equation, [c], f_tol=1e-14)
    d = norm.cdf(c - a, c, s)

    # Jesli solver wybral ujemna wartosc a nalezy przerwac dzialanie programu
    # assert(a >= 0)

    return a


def tests():
    mu, sigma = 0, 0.1

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
    y = async.map(lambda v: scipy.optimize.broyden2(func, [1./h], f_tol=1e-14), x)
    # y = [scipy.optimize.broyden1(func, [v], f_tol=1e-14) for v in x]
    # y = [func(val) for val in x]
    # plt.plot(x, y)
    # plt.grid(True)
    # plt.show()
    # print(y[0], y[-1])


def main():
    c = 10
    s = 0.1
    h = 0.1
    a = calculate_single_triangle(c, s, h)
    print(a)
    # h = 0.1

    def func(x, h, c, s):
        return norm.cdf(c - x, c, s) - 0.5 * (1 - x * h)

    print(func(a, h, c, s))
    x = np.linspace(-10, 10, 1000)
    for h in np.linspace(0.1, 2, 5):
        y = async.map(lambda v: func(v, h, c, s), x)
        plt.plot(x, y, label="%s" % h)

    plt.ylabel("Wartosc funkcji")
    plt.xlabel("Argument rownania")
    plt.title("Wykres dla c = %s, s = %s" % (c, s))
    plt.legend()
    plt.grid(True)
    plt.show()
    # print(y)


if __name__ == '__main__':
    main()
