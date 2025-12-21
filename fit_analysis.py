import numpy as np
from utils import *


def classify(fit, pos_expected, duration):
    """Assign a type to a fitted function based on what features are visible.

    The different types:
    0: first step expected direction, all steps considerable.
    1: first step expected direction, third step negligible.
    2: first step expected direction, second step negligible.
    3: first step expected direction, second and third steps negligible.
    4: first step expected direction but negligible, other steps considerable.
    5: first step expected direction but negligible, third step also negligible.
    6: first step expected direction but negligible, second step also negligible.
    7: first step expected direction but all steps negligible.
    8: first step unexpected direction, all steps considerable.
    9: first step unexpected direction, third step negligible.
    10: first step unexpected direction, second step negligible.
    11: first step unexpected direction, second and third steps negligible.
    12: first step unexpected direction but negligible, other steps considerable.
    13: first step unexpected direction but negligible, third step also negligible.
    14: first step unexpected direction but negligible, second step also negligible.
    15: first step unexpected direction but all steps negligible.

    Arguments:
    fit: a list of parameters of n_shape_margins function other than x.
    pos_expected: False if a drop of value during arousal is expected (amplitude), True if rise is expected (frequency).
    duration: The duration of the arousal in seconds.

    Returns: int corresponding to the type, NaN if fit contains NaN.
    """

    if np.any(np.isnan(np.array(fit))):
        return np.nan
    else:
        unexpected = (fit[1] > 0) != pos_expected
        start_time = -before_arousal+2
        end_time = duration+before_arousal-2
        step_times = [fit[4]]
        step_times.append(step_times[0]+1+fit[5]*(end_time-step_times[0]-3))
        step_times.append(step_times[1]+1+fit[6]*(end_time-step_times[1]-2))
        # step_times_extended = [start_time] + step_times + [end_time]
        absolute_magnitudes = [abs(fit[1]), abs(fit[2]*fit[1]), abs(fit[3]*fit[1])]
        # absolute_magnitudes_extended = [max(absolute_magnitudes)*2] + absolute_magnitudes + [max(absolute_magnitudes)*2]
        # too_close_to_bigger = [np.any([abs(step_times_extended[j] - step_times_extended[k]) < 1 and absolute_magnitudes_extended[k] > absolute_magnitudes_extended[j] for k in [0, 1, 2, 3, 4]]) for j in [1, 2, 3]]
        t = np.linspace(start_time, end_time, 1000)
        f = n_shape_margins_unrounded(t, *fit)
        meanvalue = np.mean(f)
        relative_magnitudes = [magn/meanvalue for magn in absolute_magnitudes]
        negligible = [not (relative_magnitudes[i] > 0.05) for i in [0, 1, 2]] # [too_close_to_bigger[i] or not (relative_magnitudes[i] > 0.05) for i in [0, 1, 2]]  # [not ((-before_arousal+2+1 < step_times[i] < duration+before_arousal-2-1) and (relative_magnitudes[i] > 0.05)) for i in [0, 1, 2]]
        if negligible[2] and not negligible[1] and (np.mean(f[t > step_times[1]]) - np.mean(f[t < step_times[1]])) * fit[1] > 0:
            # Two negligible steps, or even one if the steps are close to the end, are in rare cases enough to counteract a barely non-negligible step.
            negligible[1] = True
        if not np.any(negligible):
            if fit[2] < 0.5 and fit[3] > fit[2] and absolute_magnitudes[1]*(step_times[2]-step_times[1])/abs(np.sum(f - fit[0])) < 0.05:
                # a small shift towards baseline in the middle of a large and increasing change should not be considered a return to baseline and the large change should not be divided into two.
                negligible = [False, True, False]
        if negligible == [False, True, False]:
            if np.diff(np.diff(step_times))[0] < 0:  # second step closer to the third one than the first one:
                change_time = step_times[0]*fit[1]/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(-fit[1]*fit[2]+fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
            else:
                change_time = step_times[0]*(fit[1]-fit[1]*fit[2])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
            if (np.mean(f[t > change_time]) - np.mean(f[t < change_time])) * fit[1] < 0:
                # Even though the second step is deemed negligible based on its magnitude, the time between second and third steps might be long compared to the times before first step and after third step, leading to the average change being opposite to expected. These cases are extremely rare, and for simplicity all steps are considered negligible in these cases.
                negligible = [True, True, True]

        return unexpected*8 + negligible[0]*4 + negligible[1]*2 + negligible[2]


def meanfitvalue(fit, duration):
    if np.any(np.isnan(np.array(fit))):
        return np.nan
    else:
        start_time = -before_arousal+2
        end_time = duration+before_arousal-2
        step_times = [fit[4]]
        t = np.linspace(start_time, end_time, 1000)
        f = n_shape_margins_unrounded(t, *fit)
        return(np.mean(f))


def startbaseline(fit, duration, classification):
    start_time = -before_arousal+2
    end_time = duration+before_arousal-2
    step_times = [fit[4]]
    step_times.append(step_times[0]+1+fit[5]*(end_time-step_times[0]-3))
    step_times.append(step_times[1]+1+fit[6]*(end_time-step_times[1]-2))
    t = np.linspace(start_time, end_time, 1000)
    f = n_shape_margins_unrounded(t, *fit)
    if classification in [0, 1, 3, 8, 9, 11]:
        return fit[0]
    elif classification in [4, 5, 12, 13]:
        if step_times[0] >= start_time + 1:
            return np.mean(f[t < step_times[1]])
        else:
            return fit[0] + fit[1]
    elif classification in [6, 14]:
        if step_times[0] > start_time + 1:
            return np.mean(f[t < step_times[2]])
        elif step_times[1] > start_time + 1:
            return np.mean(f[(t > step_times[0]) & (t < step_times[2])])
        else:
            return fit[0] + fit[1] - fit[1]*fit[2]
    elif classification in [2, 10]:
        if np.diff(np.diff(step_times))[0] < 0:  # second step closer to the third one than the first one:
            change_time = step_times[0]*fit[1]/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(-fit[1]*fit[2]+fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
        else:
            change_time = step_times[0]*(fit[1]-fit[1]*fit[2])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
        return np.mean(f[t < change_time])
    elif classification in [7, 15]:
        return np.mean(f)
    else:
        return np.nan


startlevel = startbaseline


def endbaseline(fit, duration, classification):
    start_time = -before_arousal+2
    end_time = duration+before_arousal-2
    step_times = [fit[4]]
    step_times.append(step_times[0]+1+fit[5]*(end_time-step_times[0]-3))
    step_times.append(step_times[1]+1+fit[6]*(end_time-step_times[1]-2))
    baseline = startbaseline(fit, duration, classification)
    t = np.linspace(start_time, end_time, 1000)
    f = n_shape_margins_unrounded(t, *fit)
    meanvalue = np.mean(f)
    if classification in [1, 9]:
        if step_times[2] > end_time - 1 and not (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05:
            return fit[0] + fit[1] - fit[2]*fit[1]
        elif step_times[2] <= end_time - 1 and not (baseline - np.mean(f[t > step_times[1]]))*np.sign(fit[1])/meanvalue > 0.05:
            return np.mean(f[t > step_times[1]])
        else:
            return np.nan
    elif classification in [0, 4, 8, 12] and not (fit[0]+fit[1]-fit[2]*fit[1]+fit[3]*fit[1]-baseline)*np.sign(fit[1])/meanvalue > 0.05:
        return fit[0] + fit[1] - fit[2]*fit[1] + fit[3]*fit[1]
    elif classification in [7, 15]:
        return np.mean(f)
    else:  # 2, 3, 5, 6, 10, 11, 13, 14, NaN
        return np.nan


def endlevel(fit, duration, classification):
    # Only for data normalization. No interpretations should be made from this function's output only.
    start_time = -before_arousal+2
    end_time = duration+before_arousal-2
    step_times = [fit[4]]
    step_times.append(step_times[0]+1+fit[5]*(end_time-step_times[0]-3))
    step_times.append(step_times[1]+1+fit[6]*(end_time-step_times[1]-2))
    baseline = startbaseline(fit, duration, classification)
    t = np.linspace(start_time, end_time, 1000)
    f = n_shape_margins_unrounded(t, *fit)
    if classification in [3, 11]:
        return np.mean(f[t > step_times[0]])
    elif classification in [1, 5, 9, 13]:
        return np.mean(f[t > step_times[1]])
    elif classification in [0, 4, 6, 8, 12, 14]:
        return fit[0] + fit[1] - fit[2]*fit[1] + fit[3]*fit[1]
    elif classification in [7, 15]:
        return np.mean(f)
    elif classification in [2, 10]:
        if np.diff(np.diff(step_times))[0] < 0:  # second step closer to the third one than the first one:
            change_time = step_times[0]*fit[1]/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(-fit[1]*fit[2]+fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
        else:
            change_time = step_times[0]*(fit[1]-fit[1]*fit[2])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
        return np.mean(f[t > change_time])
    else:  # NaN
        return np.nan


def expected_change(fit, duration, classification):
    """Characterize the change to the expected direction.

    Returns:
    top_value: the signal value between the change and return. NaN if no change.
    change_time: the time of the initial change, relative to arousal start. NaN if no change.
    return_time_s: the time of return, relative to arousal start. NaN if no change or no return is detected.
    return_time_e: the time of return, relative to arousal end. NaN if no change or no return is detected.
    beforeafter: 1 if preceded by change to opposite direction, -1 if followed by such a change, 0 if alone. NaN if no change.
    """

    start_time = -before_arousal+2
    end_time = duration+before_arousal-2
    step_times = [fit[4]]
    step_times.append(step_times[0]+1+fit[5]*(end_time-step_times[0]-3))
    step_times.append(step_times[1]+1+fit[6]*(end_time-step_times[1]-2))
    t = np.linspace(start_time, end_time, 1000)
    f = n_shape_margins_unrounded(t, *fit)
    meanvalue = np.mean(f)
    baseline = startbaseline(fit, duration, classification)
    if classification == 0:
        return [fit[0] + fit[1],
                step_times[0],
                step_times[1],
                step_times[1] - duration,
                -1 if (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05 else 0]
    elif classification == 1:
        return [fit[0] + fit[1],
                step_times[0],
                step_times[1],
                step_times[1] - duration,
                -1 if ((step_times[2] > end_time - 1 and (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05) or (step_times[2] <= end_time - 1 and (baseline - np.mean(f[t > step_times[1]]))*np.sign(fit[1])/meanvalue > 0.05)) else 0]
    elif classification == 2:
        # the two changes are considered one, with weighted average of the times as time.
        if np.diff(np.diff(step_times))[0] < 0:  # second step closer to the third one than the first one:
            change_time = step_times[0]*fit[1]/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(-fit[1]*fit[2]+fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
        else:
            change_time = step_times[0]*(fit[1]-fit[1]*fit[2])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
        return [np.mean(f[t > change_time]),
                change_time,
                np.nan,
                np.nan,
                0]
    elif classification == 3:
        return [np.mean(f[t > step_times[0]]),
                step_times[0],
                np.nan,
                np.nan,
                0]
    elif classification == 4 and (fit[0]+fit[1]-fit[2]*fit[1]+fit[3]*fit[1]-baseline)*np.sign(fit[1])/meanvalue > 0.05:
        return [fit[0] + fit[1] - fit[2]*fit[1] + fit[3]*fit[1],
                step_times[2],
                np.nan,
                np.nan,
                1]
    elif classification == 6:
        return [fit[0] + fit[1] - fit[2]*fit[1] + fit[3]*fit[1],
                step_times[2],
                np.nan,
                np.nan,
                0]
    elif classification == 8 and (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05:
        return [fit[0] + fit[1] - fit[2]*fit[1],
                step_times[1],
                step_times[2],
                step_times[2] - duration,
                1]
    elif classification == 9 and ((step_times[2] > end_time - 1 and (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05) or (step_times[2] <= end_time - 1 and (baseline - np.mean(f[t > step_times[1]]))*np.sign(fit[1])/meanvalue > 0.05)):
        return [fit[0] + fit[1] - fit[2]*fit[1] if step_times[2] > end_time - 1 else np.mean(f[t > step_times[1]]),
                step_times[1],
                np.nan,
                np.nan,
                1]
    elif classification == 12:
        return [fit[0] + fit[1] - fit[1]*fit[2],
                step_times[1],
                step_times[2],
                step_times[2] - duration,
                -1 if (fit[0]+fit[1]-fit[2]*fit[1]+fit[3]*fit[1]-baseline)*np.sign(fit[1])/meanvalue > 0.05 else 0]
    elif classification == 13:
        return [fit[0] + fit[1] - fit[2]*fit[1] if step_times[2] > end_time - 1 else np.mean(f[t > step_times[1]]),
                step_times[1],
                np.nan,
                np.nan,
                0]
    else:  # 5, 7, 10, 11, 14, 15, NaN
        return [np.nan]*5


def second_expected_change(fit, duration, classification):
    """If the end state of the signal significantly differs from the baseline to the direction of the expected change, even though one such change has already occurred, characterize this second change.

    Returns:
    top_value: the signal value between the change and return. NaN if no change.
    change_time: the time of the initial change, relative to arousal start. NaN if no change.
    return_time_s: nominally the time of return, relative to arousal start. NaN as no return is actually detected.
    return_time_e: nominally the time of return, relative to arousal end. NaN as no return is actually detected.
    beforeafter: 1 if preceded by change to opposite direction, 0 if not. NaN if no change.
    """

    start_time = -before_arousal+2
    end_time = duration+before_arousal-2
    step_times = [fit[4]]
    step_times.append(step_times[0]+1+fit[5]*(end_time-step_times[0]-3))
    step_times.append(step_times[1]+1+fit[6]*(end_time-step_times[1]-2))
    t = np.linspace(start_time, end_time, 1000)
    f = n_shape_margins_unrounded(t, *fit)
    meanvalue = np.mean(f)
    baseline = startbaseline(fit, duration, classification)
    if classification == 0 and (fit[0]+fit[1]-fit[2]*fit[1]+fit[3]*fit[1]-baseline)*np.sign(fit[1])/meanvalue > 0.05:
        return [fit[0]+fit[1]-fit[2]*fit[1]+fit[3]*fit[1],  # np.mean(f[t > step_times[2]]),
                step_times[2],
                np.nan,
                np.nan,
                1 if (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05 else 0]
    else:
        return [np.nan]*5


def unexpected_change(fit, duration, classification):
    """Characterize the change to the opposite of the expected direction.

    Returns:
    top_value: the signal value between the change and return. NaN if no change.
    change_time: the time of the initial change, relative to arousal start. NaN if no change.
    return_time_s: the time of return, relative to arousal start. NaN if no change or no return is detected.
    return_time_e: the time of return, relative to arousal end. NaN if no change or no return is detected.
    beforeafter: 1 if preceded by change to opposite direction, -1 if followed by such a change, 0 if alone.
    """

    start_time = -before_arousal+2
    end_time = duration+before_arousal-2
    step_times = [fit[4]]
    step_times.append(step_times[0]+1+fit[5]*(end_time-step_times[0]-3))
    step_times.append(step_times[1]+1+fit[6]*(end_time-step_times[1]-2))
    t = np.linspace(start_time, end_time, 1000)
    f = n_shape_margins_unrounded(t, *fit)
    meanvalue = np.mean(f)
    baseline = startbaseline(fit, duration, classification)
    if classification == 0 and (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05:
        return [fit[0] + fit[1] - fit[2]*fit[1],
                step_times[1],
                step_times[2],
                step_times[2] - duration,
                1]
    elif classification == 1 and ((step_times[2] > end_time - 1 and (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05) or (step_times[2] <= end_time - 1 and (baseline - np.mean(f[t > step_times[1]]))*np.sign(fit[1])/meanvalue > 0.05)):
        return [fit[0] + fit[1] - fit[2]*fit[1] if step_times[2] > end_time - 1 else np.mean(f[t > step_times[1]]),
                step_times[1],
                np.nan,
                np.nan,
                1]
    elif classification == 4:
        return [fit[0] + fit[1] - fit[1]*fit[2],
                step_times[1],
                step_times[2],
                step_times[2] - duration,
                -1 if (fit[0]+fit[1]-fit[2]*fit[1]+fit[3]*fit[1]-baseline)*np.sign(fit[1])/meanvalue > 0.05 else 0]
    elif classification == 5:
        return [fit[0] + fit[1] - fit[2]*fit[1] if step_times[2] > end_time - 1 else np.mean(f[t > step_times[1]]),
                step_times[1],
                np.nan,
                np.nan,
                0]
    elif classification == 8:
        return [fit[0] + fit[1],
                step_times[0],
                step_times[1],
                step_times[1] - duration,
                -1 if (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05 else 0]
    elif classification == 9:
        return [fit[0] + fit[1],
                step_times[0],
                step_times[1],
                step_times[1] - duration,
                -1 if ((step_times[2] > end_time - 1 and (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05) or (step_times[2] <= end_time - 1 and (baseline - np.mean(f[t > step_times[1]]))*np.sign(fit[1])/meanvalue > 0.05)) else 0]
    elif classification == 10:
        # the two changes are considered one, with weighted average of the times as time.
        if np.diff(np.diff(step_times))[0] < 0:  # second step closer to the third one than the first one:
            change_time = step_times[0]*fit[1]/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(-fit[1]*fit[2]+fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
        else:
            change_time = step_times[0]*(fit[1]-fit[1]*fit[2])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1]) + step_times[2]*(fit[1]*fit[3])/(fit[1]-fit[2]*fit[1]+fit[3]*fit[1])
        return [np.mean(f[t > change_time]),
                change_time,
                np.nan,
                np.nan,
                0]
    elif classification == 11:
        return [np.mean(f[t > fit[4]]),
                step_times[0],
                np.nan,
                np.nan,
                0]
    elif classification == 12 and (fit[0]+fit[1]-fit[2]*fit[1]+fit[3]*fit[1]-baseline)*np.sign(fit[1])/meanvalue > 0.05:
        return [fit[0] + fit[1] - fit[2]*fit[1] + fit[3]*fit[1],
                step_times[2],
                np.nan,
                np.nan,
                1]
    elif classification == 14:
        return [fit[0] + fit[1] - fit[2]*fit[1] + fit[3]*fit[1],
                step_times[2],
                np.nan,
                np.nan,
                0]
    else:  # 2, 3, 6, 7, 13, 15, NaN
        return [np.nan]*5


def second_unexpected_change(fit, duration, classification):
    """If the end state of the signal significantly differs from the baseline to the direction of the unexpected change, even though one such change has already occurred, characterize this second change.

    Returns:
    top_value: the signal value between the change and return. NaN if no change.
    change_time: the time of the initial change, relative to arousal start. NaN if no change.
    return_time_s: nominally the time of return, relative to arousal start. NaN as no return is actually detected.
    return_time_e: nominally the time of return, relative to arousal end. NaN as no return is actually detected.
    beforeafter: 1 if preceded by change to opposite direction, 0 if not. NaN if no change.
    """

    start_time = -before_arousal+2
    end_time = duration+before_arousal-2
    step_times = [fit[4]]
    step_times.append(step_times[0]+1+fit[5]*(end_time-step_times[0]-3))
    step_times.append(step_times[1]+1+fit[6]*(end_time-step_times[1]-2))
    t = np.linspace(start_time, end_time, 1000)
    f = n_shape_unrounded(t, *fit)
    meanvalue = np.mean(f)
    baseline = startbaseline(fit, duration, classification)
    if classification == 8 and (fit[0]+fit[1]-fit[2]*fit[1]+fit[3]*fit[1]-baseline)*np.sign(fit[1])/meanvalue > 0.05:
        return [fit[0]+fit[1]-fit[2]*fit[1]+fit[3]*fit[1],  # np.mean(f[t > step_times[2]]),
                step_times[2],
                np.nan,
                np.nan,
                1 if (abs(fit[2]*fit[1])-abs(fit[1]))/meanvalue > 0.05 else 0]
    else:
        return [np.nan]*5