import numpy as np
from scipy.ndimage import gaussian_filter1d
from roerich.change_point import ChangePointDetectionClassifier
from Utils.detection import detect
import warnings

_SHARED_SERIE = None


def init_worker_kliep(shared_array, length):
    """
    Inicializa el worker compartiendo la serie de tiempo
    """
    global _SHARED_SERIE
    _SHARED_SERIE = np.frombuffer(shared_array, dtype=np.float64)
    _SHARED_SERIE = _SHARED_SERIE[:length]


def evaluate_window_worker_kliep(task):
    """
    Evalúa la función de costo para cada tamaño de ventana usando ChangePointDetectionClassifier.
    Retorna: (cost, penalty, w, change_points, distancias) si penal=True
             (cost, w, change_points, distancias) si penal=False
    """
    from Algoritmo_KLIEP.kliep_cpd import KLIEP_CPD

    w, penal, lambda_p, config = task

    serie = _SHARED_SERIE

    # Crear instancia del modelo
    model = KLIEP_CPD(
        serie,
        window=w,
        base_classifier=config["base_classifier"],
        metric=config["metric"])


    # Detectar puntos de cambio
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter("always")

        change_points = model.distancias(return_cps=True)
        distancias = model.last_distance_raw.copy()

    # Incluir bordes de la serie
    cp_full = np.concatenate(([0], change_points, [len(serie)]))

    if penal:
        total, penalty = model.total_cost(cp_full, penal=penal)
        return total, penalty, w, change_points, distancias
    else:
        cost = model.total_cost(cp_full, penal=penal)
        return cost, w, change_points, distancias
