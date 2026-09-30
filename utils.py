import datetime
from multiprocessing import Process, Manager

import numpy as np
import scipy.stats
from scipy.special import ndtri
from tqdm.notebook import tqdm, trange

def stage2num(stage, labels=[['W', 'U'], 'R', 'N1', 'N2', 'N3']):
    if stage in labels:
        return -labels.index(stage)
    elif any([stage in label for label in labels]):
        return -[stage in label for label in labels].index(True)
    else:
        return 1


def duration_from_str(duration):
    if duration.count(':') == 0:
        raise Exception(duration + 'does not look like a time duration, as it does not have a colon!')
    elif duration.count(':') > 1:
        raise Exception(duration + 'has multiple colons. Looks like its time to implement longer durations!')
    else:
        i = duration.find(':')
        return datetime.timedelta(minutes=int(duration[:i]), seconds=float(duration[i+1:])).total_seconds()


# Signal models based on logistic functions, i.e. rounded steps

def n_shape_margins(x, a, b1, bx2, bx3, c1, cx2, cx3):
    # Three consecutive logistic functions, the middle one to the opposite direction than the others, the magnitudes of the latter two defined relative to the first one, and the timing of the second
    # and the third step defined as fractions of the time left, so that they are not put within one second of other steps or end.
    return a + b1/(1+np.e**(-5*(x-c1))) - b1*bx2/(1+np.e**(-5*(x-(c1+1+cx2*(max(x)-c1-3))))) + b1*bx3/(1+np.e**(-5*(x-((c1+1+cx2*(max(x)-c1-3))+1+cx3*(max(x)-(c1+1+cx2*(max(x)-c1-3))-2)))))

def n_shape_margins_unrounded(x, a, b1, bx2, bx3, c1, cx2, cx3):
    # Like n_shape_margins, but Heaviside step functions instead of logistic functions.
    return a + b1*np.heaviside(x-c1, 0.5) - b1*bx2*np.heaviside(x-(c1+1+cx2*(max(x)-c1-3)), 0.5) + b1*bx3*np.heaviside(x-((c1+1+cx2*(max(x)-c1-3))+1+cx3*(max(x)-(c1+1+cx2*(max(x)-c1-3))-2)), 0.5)


def iterative_wilcoxon_old(x, y, x_labels, y_labels, alternative, iterations):
    p = []
    pmedians = []
    T = []
    Tmedians = []
    z = []
    zmedians = []
    for i in trange(iterations):
        x_sample = []
        y_sample = []
        for label in np.unique(x_labels):
            x_individual = [x[i] for i in range(len(x)) if x_labels[i] == label]
            y_individual = [y[i] for i in range(len(y)) if y_labels[i] == label]
            n = min(len(x_individual), len(y_individual))
            x_sample += list(np.random.choice(x_individual, size=n, replace=False))
            y_sample += list(np.random.choice(y_individual, size=n, replace=False))
        w = scipy.stats.wilcoxon(x_sample, y_sample, alternative=alternative, method='approx')
        p.append(w.pvalue)
        pmedians.append(np.median(p))
        T.append(w.statistic)
        Tmedians.append(np.median(T))
        z.append(w.zstatistic)
        zmedians.append(np.median(z))
    return np.median(p), pmedians, p, np.median(T), Tmedians, T, np.median(z), zmedians, z

def wilcoxon_iteration(x, y, x_labels, y_labels, alternative, p, T, z):
    x_sample = []
    y_sample = []
    for label in np.unique(x_labels):
        x_individual = [x[i] for i in range(len(x)) if x_labels[i] == label]
        y_individual = [y[i] for i in range(len(y)) if y_labels[i] == label]
        n = min(len(x_individual), len(y_individual))
        x_sample += list(np.random.choice(x_individual, size=n, replace=False))
        y_sample += list(np.random.choice(y_individual, size=n, replace=False))
    w = scipy.stats.wilcoxon(x_sample, y_sample, alternative=alternative, method='approx')
    p.append(w.pvalue)
    T.append(w.statistic)
    z.append(w.zstatistic)

def iterative_wilcoxon(x, y, x_labels, y_labels, alternative, iterations):
    with Manager() as manager:
        p = manager.list()
        T = manager.list()
        z = manager.list()
        processes = []
        for i in trange(iterations):
            pr = Process(target=wilcoxon_iteration, args=(x, y, x_labels, y_labels, alternative, p, T, z))
            pr.start()
            processes.append(pr)
        for pr in tqdm(processes):
            pr.join()
            pr.close()
        p = list(p)
        T = list(T)
        z = list(z)
    return np.median(p), p, np.median(T), T, np.median(z), z

def iterative_wilcoxon_unlabeled_old(x, y, alternative, iterations):
    p = []
    pmedians = []
    T = []
    Tmedians = []
    z = []
    zmedians = []
    for i in trange(iterations):
        n = min(len(x), len(y))
        x_sample = list(np.random.choice(x, size=n, replace=False))
        y_sample = list(np.random.choice(y, size=n, replace=False))
        w = scipy.stats.wilcoxon(x_sample, y_sample, alternative=alternative, method='approx')
        p.append(w.pvalue)
        pmedians.append(np.median(p))
        T.append(w.statistic)
        Tmedians.append(np.median(T))
        z.append(w.zstatistic)
        zmedians.append(np.median(z))
    return np.median(p), pmedians, p, np.median(T), Tmedians, T, np.median(z), zmedians, z

def wilcoxon_unlabeled_iteration(x, y, alternative, p, T, z):
    n = n = min(len(x), len(y))
    x_sample = list(np.random.choice(x, size=n, replace=False))
    y_sample = list(np.random.choice(y, size=n, replace=False))
    w = scipy.stats.wilcoxon(x_sample, y_sample, alternative=alternative, method='approx')
    p.append(w.pvalue)
    T.append(w.statistic)
    z.append(w.zstatistic)

def iterative_wilcoxon_unlabeled(x, y, alternative, iterations):
    with Manager() as manager:
        p = manager.list()
        T = manager.list()
        z = manager.list()
        processes = []
        for i in trange(iterations):
            pr = Process(target=wilcoxon_unlabeled_iteration, args=(x, y, alternative, p, T, z))
            pr.start()
            processes.append(pr)
        for pr in tqdm(processes):
            pr.join()
            pr.close()
        p = list(p)
        T = list(T)
        z = list(z)
    return np.median(p), p, np.median(T), T, np.median(z), z

def mwu_sumT(x, y):
    # Careful! This works for lists but not for NumPy arrays!
    data = x + y
    _,  counts = np.unique(data, return_counts=True)
    return sum([T for T in counts if T > 1])

def mannwhitneyu_to_z(U, n1, n2, sumT):
    mu = n1*n2/2
    N = n1 + n2
    sigma = np.sqrt( (n1*n2 / (N * (N - 1))) * ((N**3 - N) / 12 - sumT) )
    return (U - mu) / sigma

def effect_size_r_d(z, n1, n2):
    r = z / np.sqrt(n1 + n2)
    d = abs(2 * r / np.sqrt(1 + r**2))
    return r, d

def quantile_CI(data, quantile, confidence):
    b = scipy.stats.binom(len(data), quantile)
    q = 1 - confidence
    k = b.ppf(q=q)
    data2 = sorted(data)
    CI_lo = data2[int(k) - 1]
    CI_hi = data2[int(len(data) - k) - 1]
    return CI_lo, CI_hi

def median_difference_CI(data1, data2, confidence):
    lo1, hi1 = quantile_CI(data1, 0.5, confidence)
    lo2, hi2 = quantile_CI(data2, 0.5, confidence)
    # If the data are not independent of each other, this will overestimate the uncertainty.
    # That's more honest than underestimating it.
    return lo1 - hi2, hi1 - lo2

def IQR_CI(data, confidence):
    lo1, hi1 = quantile_CI(data, 0.25, confidence)
    lo3, hi3 = quantile_CI(data, 0.75, confidence)
    # If the data are not independent of each other, this will overestimate the uncertainty.
    # That's more honest than underestimating it.
    return lo3 - hi1, hi3 - lo1

sample_rate = 64
before_arousal = 10
