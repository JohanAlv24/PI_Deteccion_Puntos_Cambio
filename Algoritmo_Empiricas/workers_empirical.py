import numpy as np
from scipy.ndimage import gaussian_filter1d
from Utils.detection import detect

_SHARED_SERIE = None


#Función utilizada para la paralelización
def init_worker_empirical(shared_array, length):

    global _SHARED_SERIE

    _SHARED_SERIE = np.frombuffer(shared_array, dtype=np.float64)
    _SHARED_SERIE = _SHARED_SERIE[:length]

#Hace la evaluación de la función de coste para cada w por fuerza bruta
def evaluate_window_worker(task):
    from Empirical_CPD import EmpiricalCPD


    w, penal, lambda_p, config = task

    serie = _SHARED_SERIE

    model = EmpiricalCPD(
        serie,
        window=w,
        sigma=config["sigma_filter"],
        k_gauss=config["k_gauss"]
    )

    distancias = model.distancias()

    change_points = detect(distancias, w, alpha=0.05, thr=0)

    cp_full = np.concatenate(([0], change_points, [len(serie)]))
    if penal:
        total, penalty = model.total_cost(
                                            cp_full,
                                            penal=penal
                                        )
        return total, penalty,  w, change_points, distancias 
    else:
        cost = model.total_cost(cp_full, penal=penal)
        return cost, w, change_points, distancias