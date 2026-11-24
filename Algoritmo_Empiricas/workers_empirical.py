import numpy as np
from Utils.detection import detect

_SHARED_SERIE = None


# Función utilizada para la paralelización
def init_worker_empirical(
    shared_array,
    length
):
    global _SHARED_SERIE

    _SHARED_SERIE = np.frombuffer(
        shared_array,
        dtype=np.float64
    )[:length]


# Hace la evaluación de la función de coste
# para cada w por fuerza bruta
def evaluate_window_worker(task):
    from Algoritmo_Empiricas.Empirical_CPD import (
        EmpiricalCPD
    )

    (
        w,
        penal,
        config
    ) = task

    global _SHARED_SERIE
    serie = _SHARED_SERIE

    model = EmpiricalCPD(
        serie,
        window=int(w),
        smooth=config["smooth"],
        max_bandwidth=config["max_bandwidth"],
        prominence_ratio=config["prominence_ratio"]
    )

    distancias = model.distancias()

    # Es indispensable indicar que se utiliza
    # el detector empírico.
    change_points = detect(
        distancias,
        int(w),
        thr=0,
        prominence_ratio=config["prominence_ratio"]
    )

    if penal:
        total, penalty = model.total_cost(
            change_points,
            penal=True
        )

        return (
            float(total),
            int(penalty),
            int(w)
        )

    total = model.total_cost(
        change_points,
        penal=False
    )

    return (
        float(total),
        int(w)
    )