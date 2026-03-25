import numpy as np
from scipy.signal import find_peaks

def detect(distancias, window, alpha=0.05, thr=0):
    distancias = np.asarray(distancias)
    n = distancias.size
    if n == 0:
        return np.array([], dtype=int)
    if n == 1:
        return np.array([], dtype=int)

    cota = np.percentile(distancias, 100 * alpha)

    peaks, _ = find_peaks(distancias)
    mins, _  = find_peaks(-distancias)   

    # Considerar los extremos como picos o valles si son locales
    if n >= 2:
        if distancias[0] > distancias[1]:
            peaks = np.unique(np.concatenate([peaks, [0]]))
        elif distancias[0] < distancias[1]:
            mins = np.unique(np.concatenate([mins, [0]]))
        # extremo derecho
        if distancias[-1] > distancias[-2]:
            peaks = np.unique(np.concatenate([peaks, [n-1]]))
        elif distancias[-1] < distancias[-2]:
            mins = np.unique(np.concatenate([mins, [n-1]]))

    peaks = np.sort(peaks)
    mins = np.sort(mins)

    if peaks.size == 0 or mins.size == 0:
        return np.array([], dtype=int)

    changepoints = []
    for p in peaks:
        left_mins = mins[mins < p]
        right_mins = mins[mins > p]

        diffs = []
        if left_mins.size > 0:
            left_index = left_mins[-1]
            diffs.append(distancias[p] - distancias[left_index])
        if right_mins.size > 0:
            right_index = right_mins[0]
            diffs.append(distancias[p] - distancias[right_index])

        if len(diffs) == 0:
            continue

        max_diff = max(diffs)
        if max_diff >= cota and distancias[p] >= thr:
            changepoints.append(p)

    return np.array(changepoints, dtype=int) + window