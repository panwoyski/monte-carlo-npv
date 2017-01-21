import numpy as np
from tools import async
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.optimize
import math
import itertools

'''
[1] "THE NPV CRITERION FOR VALUING INVESTMENTS UNDER UNCERTAINTY" - Daniel ARMEANU
'''

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
    roots = scipy.optimize.broyden1(equation, [c - 1/h * 1.1, c, c + 1/h * 1.1], f_tol=1e-14)
    a = roots.max()
    d = norm.cdf(c - a, c, s)
    print(a, 'cdf(%s) = %s' % (c, d))


    # Jesli solver wybral ujemna wartosc a nalezy przerwac dzialanie programu
    if a < 0:
        raise RuntimeError("a value below 0, %s. (c, s, h) = (%s, %s, %s)" % (a, c, s, h))

    return (c - a, c, c + a), d


def get_equation_plots():
    def func(x, h, c, s):
        return norm.cdf(c - x, c, s) - 0.5 * (1 - x * h)
    c = 0
    s = 0.1
    h = 0.5
    a = calculate_single_triangle(c, s, h)
    print(a)
    print(func(a, h, c, s))

    x = np.linspace(-1, 1, 100)
    f, ax = plt.subplots(4, sharex=True)
    for i, s in enumerate([0.1, 0.3, 0.5, 0.7]):
        for h in np.linspace(0, 2, 5):
            y = async.map(lambda v: func(v, h, c, s), x)
            ax[i].plot(x, y, label="%s" % h)

        ax[i].set_ylim([-1, 1])
        ax[i].set_title("Wykres dla c = %s, s = %s" % (c, s))
        ax[i].legend()
        ax[i].grid(True)
    plt.show()


def generate_cash_flow(triplets, choices):
    '''
    Funckja generujaca cash flow na podstawie wyniku dzialania generatora trojkatnego
    oraz wybranej kombinacji
    :param triplets: lista wyjsc z generatora, N elementow postaci
                     ((c-a, c, c+a), d)
    :param choices: lista indeksow dla kazdego z N okresow wg klucza:
                    0 -> c-a, d
                    1 -> c, 1-2d
                    2 -> c+a, 1+2d
    :return: lista cash_flow dla n okresow z elementami postaci
             (fcf, prawdopodobienstwo czastkowe)
    '''
    fcf_p_tuples = []
    for index, choice in enumerate(choices):
        triplet, d = triplets[index]
        fcf = triplet[choice]
        p = (1 - 2 * d) if (choice == 1) else d
        fcf_p_tuples.append((fcf, p))
    return fcf_p_tuples


def calculate_npv(cash_flows, k, i0):
    '''
    Funkcja liczaca cash flow na podstawie wzorow zaczerpnietych
    z [1] wzory "I" oraz "2"
    :param cash_flows: lista cash flows dla N okresow postaci
           (cash_flow, prawdopodobienstwo_czastkowe)
    :param k: wspolczynnik dyskontowania npv z wzoru 2
    :param i0: koszt wstepny z wzoru 2
    :return: i-te npv oraz i-te prawdopodobienstwo
    '''
    p = 1
    npv = 0
    for index, (fcf, partial_probability) in enumerate(cash_flows):
        dcf = fcf / ((1 + k) ** index)
        npv += dcf
        p *= partial_probability
    npv -= i0
    return npv, p


def main():
    # Koszty wejsciowe
    i0 = 1000
    # Wartosc oczekiwana rozkladu normalnego
    c = 300
    # Odchylenie standardowe rozkladu normalnego
    s = math.sqrt(10)
    # Ilosc okresow branych pod uwage
    N = 5
    # Wspolczynnik dyskontowania
    k = 0.05
    # Lista wartosci h, TODO: sprawdzic czy ma byc stale
    h_list = np.linspace(0.1, 0.25, N)
    # Wygenerowane trojki oraz wartosc dystrybuanty
    # lista krotek postaci ((c-a, c, c+a), d)
    triplets = [calculate_single_triangle(c, s, h) for h in h_list]
    # from pprint import pprint
    # pprint(triplets)

    input = []
    '''
    itertools.product([0, 1, 2], repeat=N)
    generuje wszyskie mozliwe N elementowe ciagi
    z elementami z listy [0, 1, 2], w rozwazanym
    przypadku sa to indeksy z trojki wartosci zwroconej
    przez generator trojkatny
    '''
    for combination in itertools.product([0, 1, 2], repeat=N):
        cash_flow = generate_cash_flow(triplets, combination)
        npv_tuple = calculate_npv(cash_flow, k, i0)
        input.append(npv_tuple)

    # wartosc oczekiwana npv wg wzoru (2) z [1]
    mean_npv = sum([npvi * pi for npvi, pi in input])
    print(mean_npv)
    # odchylenie standardowe wg wzoru (6) z [1]
    std_dev = math.sqrt(sum([pi * ((npvi - mean_npv)**2) for npvi, pi in input]))
    print(std_dev)


if __name__ == '__main__':
    main()
