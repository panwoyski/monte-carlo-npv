import numpy as np
import scipy.optimize
import math
import itertools
from scipy.stats import norm
from timeit import default_timer as timer


def timed_execution(f):
    def decorated(*args, **kwargs):
        start = timer()
        res = f(*args, **kwargs)
        end = timer()
        exec_time = end - start
        # print('%s exec time: %.6s' % (f.__name__, exec_time))
        return res, exec_time
    return decorated


def calculate_single_triangle(c, s, h):
    if h <= 0:
        raise ValueError("h(n) musi byc wieksze od zera")

    def equation(a):
        return norm.cdf(c - a, c, s) - 0.5 * (1 - a * h)

    initial_offset = 10 * (1+s)
    initial_guesses = [-c * initial_offset, c, c * initial_offset]
    roots = scipy.optimize.broyden2(equation, initial_guesses, f_tol=1e-14)
    # print(roots)
    a = roots.max()
    d = norm.cdf(c - a, c, s)

    return c, a, d


def generate_cash_flow(triplets, choices):
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
    p = 1
    npv = 0
    for index, (fcf, partial_probability) in enumerate(cash_flows):
        dcf = fcf / ((1 + k) ** index)
        npv += dcf
        p *= partial_probability
    npv -= i0
    return npv, p


@timed_execution
def count_custom_method(cs_list, k, i0):
    N = len(cs_list)

    triplets = [calculate_single_triangle(c, s, h) for c, s, h in cs_list]

    npvs = []
    for combination in itertools.product([0, 1, 2], repeat=N):
        # print(combination)
        cash_flow = generate_cash_flow(triplets, combination)
        # print([cf for cf, _ in cash_flow])
        npv_tuple = calculate_npv(cash_flow, k, i0)
        npvs.append(npv_tuple)

    mean_npv = sum([npvi * pi for npvi, pi in npvs])
    # odchylenie standardowe wg wzoru (6) z [1]
    std_dev = np.sqrt(sum(pi * ((npvi - mean_npv)**2) for npvi, pi in npvs))
    # print('custom generator (%.4f, %.4f)' % (mean_npv, std_dev))
    return mean_npv, std_dev


def get_uniform_npv(cs_list, i0, k):
    npv = -i0
    for index, (mean, sigma, _) in enumerate(cs_list):
        fcf = np.random.normal(mean, sigma)
        dcf = fcf / ((1 + k) ** index)
        npv += dcf
    return npv


@timed_execution
def count_uniform_distribution(cs_list, k, i0):
    N = len(cs_list)
    npvs = np.array([get_uniform_npv(cs_list, i0, k) for _ in range(3 ** N)])
    normalizing_factor = npvs.shape[0]
    mean_npv = npvs.sum() / normalizing_factor
    std_dev = np.sqrt((np.power(npvs - mean_npv, 2) * 1/normalizing_factor).sum())
    # print("Unifrom distribution (%.4f, %.4f)" % (mean_npv, std_dev))
    return mean_npv, std_dev


def get_npv_from_triangle(triplets, k, i0):
    npv = -i0
    for index, (c, a, _) in enumerate(triplets):
        fcf = np.random.triangular(c - a, c, c + a)
        dcf = fcf / ((1 + k) ** index)
        npv += dcf
    return npv


@timed_execution
def count_triangle_generator(cs_list, k, i0):
    N = len(cs_list)

    triplets = [calculate_single_triangle(c, s, h) for c, s, h in cs_list]

    npvs = np.array([get_npv_from_triangle(triplets, k, i0) for _ in range(3 ** N)])

    normalizing_factor = npvs.shape[0]
    mean_npv = npvs.sum() / normalizing_factor
    std_dev = np.sqrt((np.power(npvs - mean_npv, 2) * 1/normalizing_factor).sum())

    # print("Triangle distribution (%.4f, %.4f)" % (mean_npv, std_dev))
    return mean_npv, std_dev


def get_mean_values(test_values):
    time = np.array([time for _, time in test_values]).mean()
    mean_value = np.array([mean_value for (mean_value, _), _ in test_values]).mean()
    mean_std_dev = np.array([std_dev for (_, std_dev), _ in test_values]).mean()

    return time, mean_value, mean_std_dev


def main():
    i0 = 500
    k = 0.05

    cs_list = [
        (-200, math.sqrt(0.1), 1/50),
        (-50,  math.sqrt(1),   1/50),
        (-1,   math.sqrt(10),  1/100),
        (100,  math.sqrt(100), 1/25),
        (300,  math.sqrt(200), 1/20),
        (200,  math.sqrt(250), 1/40),
        (600,  math.sqrt(90),  1/100),
    ]
    probes = 4

    from tools import async

    test_values1 = async.map(lambda _: count_custom_method(cs_list, k, i0), range(probes))
    test_values2 = async.map(lambda _: count_uniform_distribution(cs_list, k, i0), range(probes))
    test_values3 = async.map(lambda _: count_triangle_generator(cs_list, k, i0), range(probes))

    print("Sampled %s times" % probes)
    print("                 time_avg, mean avg,   std_dev avg")
    print("Custom method:   %.6f, %4.6f, %2.6f" % get_mean_values(test_values1))
    print("Uniform method:  %.6f, %4.6f, %2.6f" % get_mean_values(test_values2))
    print("Triangle method: %.6f, %4.6f, %2.6f" % get_mean_values(test_values3))

if __name__ == '__main__':
    main()
