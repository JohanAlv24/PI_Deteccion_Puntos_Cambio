from sklearn.linear_model import LogisticRegression
from roerich.change_point import ChangePointDetectionClassifier
import numpy as np

def detectar_cambios_roerich(
    serie,
    window_size=None,
    seed=42,
    periods=3
):
    serie = np.asarray(
        serie,
        dtype=np.float64
    ).ravel()


    N = serie.size

    # Regla fija del 5 % utilizada en el artículo
    # comparativo. El mínimo de 5 garantiza que el
    # posprocesamiento interno sea numéricamente factible.
    if window_size is None:
        window_size = max(
            5,
            int(round(0.05 * N))
        )
    else:
        window_size = int(window_size)


    X = serie.reshape(-1, 1)

    np.random.seed(seed)

    clasificador = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=seed
    )

    detector = ChangePointDetectionClassifier(
        base_classifier=clasificador,
        metric="klsym",
        periods=periods,
        window_size=window_size,
        step=1,
        n_runs=1
        )

    _, puntos_cambio = detector.predict(X)

    return np.asarray(
        puntos_cambio,
        dtype=int
    ).tolist()