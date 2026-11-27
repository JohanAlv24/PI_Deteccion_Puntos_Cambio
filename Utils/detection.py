import numpy as np
from scipy.signal import find_peaks, peak_prominences
from scipy.stats import median_abs_deviation, norm

'''
La función detect() recibe como entrada el vector de distancias (ya sea de Wasserstein o basado en distribuciones empíricas) 
y el tamaño de ventana utilizado. A partir de este vector identifica sus picos y luego conserva únicamente aquellos cuya altura 
supere en al menos un percentil 5 la de alguno de los valles adyacentes. Los extremos también son considerados valles o picos según 
sea el caso.
'''


def obtener_prominencias(curva):
    curva = np.asarray(
        curva,
        dtype=np.float64
    )

    if curva.size < 3:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float)
        )

    peaks, _ = find_peaks(curva)

    if peaks.size == 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float)
        )

    # Los valles son los picos de la curva negativa.
    valleys, _ = find_peaks(-curva)

    # Considerar los extremos como posibles valles.
    valleys = np.unique(
        np.concatenate([
            [0],
            valleys,
            [curva.size - 1]
        ])
    )

    prominencias = []

    for peak in peaks:
        # Valle inmediatamente anterior.
        left_valley = valleys[
            valleys < peak
        ][-1]

        # Valle inmediatamente posterior.
        right_valley = valleys[
            valleys > peak
        ][0]

        distancia_izquierda = (
            curva[peak]
            - curva[left_valley]
        )

        distancia_derecha = (
            curva[peak]
            - curva[right_valley]
        )

        # Distancia vertical al valle más lejano.
        prominencia = max(
            distancia_izquierda,
            distancia_derecha
        )

        prominencias.append(prominencia)

    return (
        peaks.astype(int),
        np.asarray(prominencias)
    )

def detect(
    distancias,
    window,
    prominence_ratio=0.05,
    thr=0,
    min_separation=0
):
    distancias = np.asarray(
        distancias,
        dtype=np.float64
    )

    window = int(window)
    min_separation = int(
                        min_separation
                    )

    if distancias.size < 3:
        return np.array([], dtype=int)

    # Prominencia redefinida como máxima distancia
    # vertical a uno de los dos valles adyacentes.
    peaks, prominences = obtener_prominencias(
        distancias
    )

    if peaks.size == 0:
        return np.array([], dtype=int)

    q05, q95 = np.quantile(
        distancias,
        [0.05, 0.95]
    )

    robust_range = q95 - q05

    if robust_range <= 0:
        return np.array([], dtype=int)

    prominence_thr = (
        prominence_ratio
        * robust_range
    )

    significant = (
        (prominences >= prominence_thr)
        & (distancias[peaks] >= thr)
    )

    peaks = peaks[significant]
    prominences = prominences[significant]


    if peaks.size == 0:
        return np.array([], dtype=int)

    if min_separation > 1 and peaks.size > 1:

        heights = distancias[peaks]

        # Prioridad:
        # 1. mayor prominencia;
        # 2. mayor altura;
        # 3. menor índice.
        order = np.lexsort(
            (
                peaks,
                -heights,
                -prominences
            )
        )

        blocked = np.zeros(
            distancias.size,
            dtype=bool
        )

        selected = []
        radius = min_separation - 1

        for index in order:
            peak = int(peaks[index])

            if blocked[peak]:
                continue

            selected.append(peak)

            left = max(
                0,
                peak - radius
            )

            right = min(
                distancias.size,
                peak + radius + 1
            )

            blocked[left:right] = True

        peaks = np.asarray(
            sorted(selected),
            dtype=int
        )

    return (
        peaks.astype(int)
        + window
    )
