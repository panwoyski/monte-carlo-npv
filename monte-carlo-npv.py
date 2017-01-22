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

        fcf = 0
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


def main():
    # Koszty wejsciowe
    i0 = 500
    # Wartosc oczekiwana rozkladu normalnego
    # c = 300
    # Odchylenie standardowe rozkladu normalnego
    # s = math.sqrt(10)
    # Ilosc okresow branych pod uwage
    N = 5
    # Wspolczynnik dyskontowania
    k = 0.05

    cs_list = [
        (-200, math.sqrt(0.1), 1/50),
        (-1, math.sqrt(1), 1/50),
        (50, math.sqrt(10), 1/100),
        (400, math.sqrt(100), 1/25),
        (500, math.sqrt(200), 1/20),
        (700, math.sqrt(250), 1/40),
        (1000, math.sqrt(90), 1/100),
    ]

    # Wygenerowane trojki oraz wartosc dystrybuanty
    # lista krotek postaci ((c-a, c, c+a), d)
    triplets = [calculate_single_triangle(c, s, h) for c, s, h in cs_list]
    from pprint import pprint
    pprint(triplets)

    test1 = [c - a for c, a, _ in triplets]
    test2 = [c for c, _, _ in triplets]
    test3 = [c + a for c, a, _ in triplets]

    plt.plot(test1, 'ro', test2, 'go', test3, 'bo')
    plt.grid()
    plt.show()

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

    print(len(set(npvs)))
    npvs = np.array(npvs)
    # wartosc oczekiwana npv wg wzoru (2) z [1]
    mean_npv = sum([npvi * pi for npvi, pi in npvs])
    print('Expected value: %s' % mean_npv)
    # odchylenie standardowe wg wzoru (6) z [1]
    std_dev = math.sqrt(sum(pi * ((npvi - mean_npv)**2) for npvi, pi in npvs))
    print('Standard deviation: %s' % std_dev)
    print_histogram(npvs)


if __name__ == '__main__':
    main()
