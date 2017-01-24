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
    Definicja rownania ktorego miejscem zerowym jest wartosc a
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
    # initial_guesses = [c - 1/h * 1.1, c, c + 1/h * 1.1]
    initial_offset = 10 * (1+s)
    initial_guesses = [-c * initial_offset, c, c * initial_offset]
    roots = scipy.optimize.broyden2(equation, initial_guesses, f_tol=1e-14)
    print(roots)
    a = roots.max()
    d = norm.cdf(c - a, c, s)
    # print(a, 'cdf(%s) = %s' % (c - a, d))

    return c, a, d


def generate_cash_flow(triplets, choices):
    """
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
    """
    fcf_p_tuples = []
    for index, choice in enumerate(choices):
        c, a, d = triplets[index]

        if choice == 0:
            fcf = c - a
        elif choice == 1:
            fcf = c
        elif choice == 2:
            fcf = c + a
        else:
            raise ValueError("Unsupported choice (%s)" % choice)

        p = (1 - 2 * d) if (choice == 1) else d
        fcf_p_tuples.append((fcf, p))
    return fcf_p_tuples


def calculate_npv(cash_flows, k, i0):
    """
    Funkcja liczaca cash flow na podstawie wzorow zaczerpnietych
    z [1] wzory "I" oraz "2"
    :param cash_flows: lista cash flows dla N okresow postaci
           (cash_flow, prawdopodobienstwo_czastkowe)
    :param k: wspolczynnik dyskontowania npv z wzoru 2
    :param i0: koszt wstepny z wzoru 2
    :return: i-te npv oraz i-te prawdopodobienstwo
    """
    p = 1
    npv = 0
    for index, (fcf, partial_probability) in enumerate(cash_flows):
        dcf = fcf / ((1 + k) ** index)
        npv += dcf
        p *= partial_probability
    npv -= i0
    return npv, p


def print_histogram(npvs):
    npv_values = [npv for npv, _ in npvs]
    print(len(set(npv_values)))

    hist, bins = np.histogram(npv_values, bins=10)
    width = 0.7 * (bins[1] - bins[0])
    center = (bins[:-1] + bins[1:]) / 2
    fig, ax = plt.subplots()
    plt.xlabel('Wartosc npv')
    plt.ylabel('Ilosc wystapien')
    ax.bar(center, hist, align='center', width=width)
    ax.set_xticks(bins)
    ax.set_title('Histogram wartosci npv dla generatora trojkatnego')
    ax.grid(True)
    plt.show()


def print_equation():
    def equation(c, s, h, a):
        return norm.cdf(c - a, c, s) - 0.5 * (1 - a * h)

    def plot_for_h_range(c, s, h_list):
        for h in h_list:
            rang = np.linspace(-1, 1, 50)
            points = async.map(lambda x: equation(c, s, h, x), rang)
            plt.plot(rang, points, label=h)
        plt.title('Wykres rownania dla c = %s, s = %s oraz zakresu wartosci h' % (c, s))
        plt.legend()
        plt.grid()
        plt.show()

    def plot_for_s_range(c, s_list, h):
        for s in s_list:
            rang = np.linspace(-1, 1, 50)
            points = async.map(lambda x: equation(c, s, h, x), rang)
            plt.plot(rang, points, label=h)
        plt.title('Wykres rownania dla c = %s, h = %s oraz zakresu wartosci s' % (c, h))
        plt.legend()
        plt.grid()
        plt.show()

    s = 200
    plot_for_h_range(0, s, [1/s, 10/s, 100/s])
    h = 1/30
    plot_for_s_range(0, [1/h, 10/h, 100/h], h)


def show_triplets_plot(triplets):
    test1 = [c - a for c, a, _ in triplets]
    test2 = [c for c, _, _ in triplets]
    test3 = [c + a for c, a, _ in triplets]

    base = [i + 1 for i in range(len(triplets))]
    plt.plot(base, test1, 'ro', label='c - a')
    plt.plot(base, test2, 'go', label='c')
    plt.plot(base, test3, 'bo', label='c + a')
    plt.title('Histogram wartosci npv dla generatora trojkatnego')
    plt.xlabel('n')
    plt.legend()
    plt.ylabel('Wygenerowane wartosci')
    plt.grid()
    plt.show()


def main():
    # Koszty wejsciowe
    i0 = 500
    # Wartosc oczekiwana rozkladu normalnego
    # c = 300
    # Odchylenie standardowe rozkladu normalnego
    # s = math.sqrt(10)
    # Wspolczynnik dyskontowania
    k = 0.05

    # Trojki c, s, h - wartosc oczekiwana, odchylenie standardowe, wysokosc trojkata
    cs_list = [
        (-200, math.sqrt(0.1), 1/50),
        (-50,  math.sqrt(1),   1/50),
        (-1,   math.sqrt(10),  1/100),
        (100,  math.sqrt(100), 1/25),
        (300,  math.sqrt(200), 1/20),
        (200,  math.sqrt(250), 1/40),
        (600,  math.sqrt(90),  1/100),
    ]
    # Ilosc okresow branych pod uwage
    N = len(cs_list)

    # Wygenerowane trojki oraz wartosc dystrybuanty
    # lista krotek postaci ((c-a, c, c+a), d)
    triplets = [calculate_single_triangle(c, s, h) for c, s, h in cs_list]
    # from pprint import pprint
    # pprint(triplets)

    show_triplets_plot(triplets)

    npvs = []
    '''
    itertools.product([0, 1, 2], repeat=N)
    generuje wszyskie mozliwe N elementowe ciagi
    z elementami z listy [0, 1, 2], w rozwazanym
    przypadku sa to indeksy z trojki wartosci zwroconej
    przez generator trojkatny
    '''
    for combination in itertools.product([0, 1, 2], repeat=N):
        # print(combination)
        cash_flow = generate_cash_flow(triplets, combination)
        # print([cf for cf, _ in cash_flow])
        npv_tuple = calculate_npv(cash_flow, k, i0)
        npvs.append(npv_tuple)

    # print(len(set(npvs)))
    npvs = np.array(npvs)
    # wartosc oczekiwana npv wg wzoru (2) z [1]
    mean_npv = sum([npvi * pi for npvi, pi in npvs])
    print('Expected value: %s' % mean_npv)
    # odchylenie standardowe wg wzoru (6) z [1]
    std_dev = math.sqrt(sum(pi * ((npvi - mean_npv)**2) for npvi, pi in npvs))
    print('Standard deviation: %s' % std_dev)
    print_histogram(npvs)


if __name__ == '__main__':
    # main()
    print_equation()