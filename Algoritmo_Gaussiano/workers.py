import numpy as np
from Utils.detection import detect

_SHARED_SERIE = None

#Función utilizada para la paralelización
def init_worker(shared_array, length):
    global _SHARED_SERIE
    _SHARED_SERIE = np.frombuffer(shared_array, dtype=np.float64)[:length]

#Hace la evaluación de la función de coste para los parámetros (w, t) por fuerza bruta

def evaluate_params_worker(args):
    from Algoritmo_Gaussiano.cpd import CPD

    (
        w,
        t,
        penal,
        class_defaults
    ) = args

    global _SHARED_SERIE
    serie = _SHARED_SERIE

    obj = CPD(
        serie,
        window=int(w),
        t=int(t),
        m=class_defaults.get(
            "m",
            3
        ),
        medias=class_defaults.get(
            "medias",
            True
        ),
        smooth=class_defaults.get(
            "smooth",
            True
        ),
        n_perm=class_defaults.get(
            "n_perm",
            0
        ),
        temporal_comp=class_defaults.get(
            "temporal_comp",
            True
        ),
        max_comp=class_defaults.get(
            "max_comp",
            False
        ),
        min_comp=class_defaults.get(
            "min_comp",
            False
        ),
        max_bandwidth=class_defaults.get(
            "max_bandwidth",
            False
        ),
        prominence_ratio=class_defaults.get(
            "prominence_ratio",
            0.05
        )
    )

    min_separation = (obj.m - 1) * obj.t + (obj.m+int(obj.temporal_comp)) + 1
    distancias = obj.distancias()

    cps = detect(
        distancias,
        obj.window,
        thr=0,
        min_separation=min_separation,
        prominence_ratio=obj.prominence_ratio
    )

    cp_full = np.concatenate(
        (
            np.array([0], dtype=int),
            np.asarray(cps, dtype=int),
            np.array(
                [len(serie)],
                dtype=int
            )
        )
    )

    if penal:
        total, penalty = obj.total_cost(
            cp_full,
            penal=True
        )

        return (
            float(total),
            float(penalty),
            int(w),
            int(t)
        )

    total = obj.total_cost(
        cp_full,
        penal=False
    )

    return (
        float(total),
        int(w),
        int(t)
    )

