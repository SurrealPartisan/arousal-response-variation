import datetime
import numpy as np
import scipy.stats
from tqdm.notebook import trange

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


def iterative_wilcoxon(x, y, x_labels, y_labels, alternative, iterations):
    p = []
    pmedians = []
    for i in trange(iterations):
        x_sample = []
        y_sample = []
        for label in np.unique(x_labels):
            x_individual = [x[i] for i in range(len(x)) if x_labels[i] == label]
            y_individual = [y[i] for i in range(len(y)) if y_labels[i] == label]
            n = min(len(x_individual), len(y_individual))
            x_sample += list(np.random.choice(x_individual, size=n, replace=False))
            y_sample += list(np.random.choice(y_individual, size=n, replace=False))
        p.append(scipy.stats.wilcoxon(x_sample, y_sample, alternative=alternative).pvalue)
        pmedians.append(np.median(p))
    return np.median(p), pmedians, p

sample_rate = 64
before_arousal = 10