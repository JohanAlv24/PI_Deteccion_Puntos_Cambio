import numpy as np
import ruptures as rpt

from scipy.signal import find_peaks


def detectar_window_rbf(
    serie,
    window=50,
    prominence_ratio=0.05
):
    """
    Window-RBF sobre la serie original, sin retardos
    y sin suavizamiento.

    La selección final utiliza prominencia estándar y
    una separación mínima determinada por window.
    """

    serie = np.asarray(
        serie,
        dtype=np.float64
    ).ravel()

    window = int(window)

    if window < 1:
        raise ValueError(
            "window debe ser un entero positivo."
        )

    if serie.size <= 2 * window:
        raise ValueError(
            "La longitud de la serie debe ser mayor "
            "que dos veces el tamaño de ventana."
        )

    if not np.all(np.isfinite(serie)):
        raise ValueError(
            "La serie contiene valores NaN o infinitos."
        )

    algorithm = rpt.Window(
        width=2 * window,
        model="rbf",
        min_size=2,
        jump=1
    ).fit(serie)

    score = np.asarray(
        algorithm.score,
        dtype=np.float64
    )

    q05, q95 = np.quantile(
        score,
        [0.05, 0.95]
    )

    robust_range = float(
        q95 - q05
    )

    if robust_range <= 0:
        return []

    prominence_threshold = (
        prominence_ratio
        * robust_range
    )

    peak_indices, _ = find_peaks(
        score,
        prominence=prominence_threshold,
        distance=window
    )

    # algorithm.inds contiene los índices originales
    # de la serie asociados con cada elemento del score.
    change_points = np.asarray(
        algorithm.inds,
        dtype=int
    )[peak_indices]

    return change_points.tolist()


