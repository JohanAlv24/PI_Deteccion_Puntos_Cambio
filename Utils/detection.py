import numpy as np
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation

'''
La función detect() recibe como entrada el vector de distancias (ya sea de Wasserstein o basado en distribuciones empíricas) 
y el tamaño de ventana utilizado. A partir de este vector identifica sus picos y luego conserva únicamente aquellos cuya altura 
supere en al menos un percentil 5 la de alguno de los valles adyacentes. Los extremos también son considerados valles o picos según 
sea el caso.
'''
def detect(distancias, window, alpha=0.05, thr=0, emp=False):

    distancias = np.asarray(distancias, dtype=np.float64)

    n = distancias.size

    if n < 3:
        return np.array([], dtype=int)

    mad = median_abs_deviation(distancias, scale='normal')

    all_prominences = np.abs(np.diff(distancias))

    prominence = np.quantile(all_prominences, 0.95)

    if emp:
        prominence = np.max(distancias)/1000
    peaks, properties = find_peaks(
        distancias,
        prominence=prominence,
        distance=window
    )

    if peaks.size == 0:
        return np.array([], dtype=int)

    mask = distancias[peaks] >= thr

    peaks = peaks[mask]

    return peaks.astype(int) + window