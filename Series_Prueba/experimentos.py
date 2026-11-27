import numpy as np
from Algoritmo_Gaussiano.cpd import CPD
from Algoritmo_Empiricas.Empirical_CPD import EmpiricalCPD
from Utils.detection import detect
from Utils.metrics_sup import metrics


#Condición de estabilidad para los rezagos
def sample_stationary_ar2():
    while True:
        phi1 = np.random.uniform(-0.9, 0.9)
        phi2 = np.random.uniform(-0.9, 0.9)

        if (abs(phi2) < 1 and
            phi1 + phi2 < 1 and
            phi2 - phi1 < 1):
            return np.array([phi1, phi2])

#Condición de estabilidad para los rezagos
def sample_invertible_ma2():
    while True:
        theta1 = np.random.uniform(-0.9, 0.9)
        theta2 = np.random.uniform(-0.9, 0.9)

        if (abs(theta2) < 1 and
            theta1 + theta2 < 1 and
            theta2 - theta1 < 1):
            return np.array([theta1, theta2])

#La función genera la serie de tiempo ARMA descrita en el artículo de revisión de métodos de detección de puntos de cambio

def ar2_noise(
    T=2000,
    n_segmentos=16,
    phi=(0.3, 0.5),
    theta=(0.0, 0.0),
    random_phi=False,
    random_theta=False,
    min_seg=50,
    max_seg=200,
    base_mean=0.0,
    random_mean=True,
    mean_range=(-1.0, 1.0),
    min_mean_diff=0.0,
    base_std=0.5,
    random_std=True,
    std_range=(0.2, 1.2),
    outlier_interval=200,
    outlier_scale=6.0,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    if not (n_segmentos * min_seg <= T <= n_segmentos * max_seg):
        raise ValueError(
            "T debe estar entre n_segmentos * min_seg y "
            "n_segmentos * max_seg."
        )

    if min_mean_diff < 0:
        raise ValueError("min_mean_diff debe ser no negativo.")

    if not random_mean and n_segmentos > 1 and min_mean_diff > 0:
        raise ValueError(
            "No puede imponerse una diferencia de medias "
            "con random_mean=False."
        )

    # Generación de longitudes
    longitudes = []
    restante = T

    for i in range(n_segmentos - 1):

        seg_restantes = n_segmentos - i - 1

        minimo_factible = max(
            min_seg,
            restante - seg_restantes * max_seg
        )

        maximo_factible = min(
            max_seg,
            restante - seg_restantes * min_seg
        )

        L = np.random.randint(
            minimo_factible,
            maximo_factible + 1
        )

        longitudes.append(L)
        restante -= L

    longitudes.append(restante)

    # Intervalo efectivo de medias
    def transformar_media(z):
        return z + np.exp(-abs(z))

    mu_min = transformar_media(mean_range[0])
    mu_max = transformar_media(mean_range[1])

    # Generación del proceso
    x = np.zeros(T)
    eps = np.zeros(T)

    cps = []
    t = 0

    phi_t = np.array(phi)
    theta_t = np.array(theta)

    media_anterior = None

    for i, L in enumerate(longitudes):

        end = t + L

        # Media del segmento
        if random_mean:
            x_mu = np.random.uniform(*mean_range)
            mu = transformar_media(x_mu)

            if media_anterior is not None:
                if abs(mu - media_anterior) < min_mean_diff:

                    mu = media_anterior + min_mean_diff

                    if mu > mu_max:
                        mu = media_anterior - min_mean_diff

                    if mu < mu_min:
                        raise ValueError(
                            "No existe una media factible para "
                            "la separación solicitada."
                        )
        else:
            mu = base_mean

        media_anterior = mu

        # Desviación estándar del segmento
        if random_std:
            sigma = np.random.uniform(*std_range)
        else:
            sigma = base_std

        if random_phi:
            phi_t = sample_stationary_ar2()

        if random_theta:
            theta_t = sample_invertible_ma2()

        eps[t:end] = np.random.normal(mu, sigma, L)

        if i > 0:
            cps.append(t)

        for j in range(max(t, 2), end):
            x[j] = (
                phi_t[0] * x[j - 1]
                + phi_t[1] * x[j - 2]
                + eps[j]
                + theta_t[0] * eps[j - 1]
                + theta_t[1] * eps[j - 2]
            )

        t = end

    # Incorporación de outliers
    outlier_idx = np.arange(
        outlier_interval, T, outlier_interval
    )

    eps[outlier_idx] += np.random.normal(
        0.0,
        outlier_scale * np.std(eps),
        size=len(outlier_idx)
    )

    x[outlier_idx] += eps[outlier_idx]

    return x, sorted(cps), outlier_idx


def generar_experimento_ar1(
    N,
    n_segmentos,
    min_sep,
    max_sep,
    betas=(0.8, -0.8),
    sigma_y=1.0,
    seed=None,
    punto_cambio=None,
    A=0
):

    rng = np.random.default_rng(seed)

    if punto_cambio is not None:
        if n_segmentos != 2:
            raise ValueError(
                "punto_cambio solo está soportado cuando n_segmentos == 2."
            )
        if not (0 < punto_cambio < N):
            raise ValueError(
                "punto_cambio debe estar estrictamente entre 0 y N."
            )

        longitudes = [punto_cambio, N - punto_cambio]

    else:

        if N < n_segmentos * min_sep or N > n_segmentos * max_sep:
            raise ValueError(
                "N debe estar entre n_segmentos * min_sep y "
                "n_segmentos * max_sep."
            )

        longitudes = []
        restante = N

        for i in range(n_segmentos - 1):

            seg_restantes = n_segmentos - i - 1

            minimo_factible = max(
                min_sep,
                restante - seg_restantes * max_sep
            )

            maximo_factible = min(
                max_sep,
                restante - seg_restantes * min_sep
            )

            L = rng.integers(
                minimo_factible,
                maximo_factible + 1
            )

            longitudes.append(L)
            restante -= L

        longitudes.append(restante)

    betas_segmentos = [rng.choice(betas)]

    for _ in range(n_segmentos - 1):

        candidatos = [
            b for b in betas
            if b != betas_segmentos[-1]
        ]

        betas_segmentos.append(
            rng.choice(candidatos)
        )

    y = np.zeros(N)

    cps = []
    segmentos = []

    pos = 0

    y[0] = rng.normal(A, sigma_y)

    for i, (L, beta) in enumerate(
        zip(longitudes, betas_segmentos)
    ):

        sigma_eps = sigma_y * np.sqrt(
            1 - beta**2
        )

        inicio = pos
        fin = pos + L

        segmentos.append({
            "segmento": i,
            "inicio": inicio,
            "fin": fin - 1,
            "longitud": L,
            "beta": beta,
            "sigma_eps": sigma_eps
        })

        if i > 0:
            cps.append(inicio)

        for t in range(max(1, inicio), fin):

            y[t] = (
                (beta * y[t - 1])
                + rng.normal(0, sigma_eps)
            )

        pos = fin

    return (
        y,
        np.array(cps),
        segmentos
    )

def generar_experimento_senoidal(
    N,
    n_segmentos,
    min_sep,
    max_sep,
    A=1.0,
    sigma_eps=0.5,
    omegas=(np.pi/10, np.pi/5),
    fase_aleatoria=False,
    seed=None,
    punto_cambio=None
):

    rng = np.random.default_rng(seed)

    if punto_cambio is not None:
        if n_segmentos != 2:
            raise ValueError(
                "punto_cambio solo está soportado cuando n_segmentos == 2."
            )
        if not (0 < punto_cambio < N):
            raise ValueError(
                "punto_cambio debe estar estrictamente entre 0 y N."
            )

        longitudes = [punto_cambio, N - punto_cambio]

    else:

        if N < n_segmentos * min_sep or N > n_segmentos * max_sep:
            raise ValueError(
                "N debe estar entre n_segmentos * min_sep y "
                "n_segmentos * max_sep."
            )

        longitudes = []
        restante = N

        for i in range(n_segmentos - 1):

            seg_restantes = n_segmentos - i - 1

            minimo_factible = max(
                min_sep,
                restante - seg_restantes * max_sep
            )

            maximo_factible = min(
                max_sep,
                restante - seg_restantes * min_sep
            )

            L = rng.integers(
                minimo_factible,
                maximo_factible + 1
            )

            longitudes.append(L)
            restante -= L

        longitudes.append(restante)

    omegas_segmentos = [rng.choice(omegas)]

    for _ in range(n_segmentos - 1):

        candidatos = [
            w for w in omegas
            if w != omegas_segmentos[-1]
        ]

        omegas_segmentos.append(
            rng.choice(candidatos)
        )

    y = np.zeros(N)

    cps = []
    segmentos = []

    pos = 0

    for i, (L, omega) in enumerate(
        zip(longitudes, omegas_segmentos)
    ):

        t_local = np.arange(L)

        if fase_aleatoria:
            phi = rng.uniform(0, 2*np.pi)
        else:
            phi = 0.0

        señal = A * np.sin(
            omega * t_local + phi
        )

        ruido = rng.normal(
            0,
            sigma_eps,
            size=L
        )

        y[pos:pos+L] = señal + ruido

        segmentos.append({
            "segmento": i,
            "inicio": pos,
            "fin": pos + L - 1,
            "longitud": L,
            "omega": omega,
            "periodo": 2*np.pi / omega,
            "fase": phi
        })

        if i > 0:
            cps.append(pos)

        pos += L

    return (
        y,
        np.array(cps),
        segmentos
    )

def generar_experimento_ar2(
    N,
    n_segmentos,
    min_sep,
    max_sep,
    sigma=0.5,
    seed=None,
    punto_cambio=None,
    A = 0
):

    rng = np.random.default_rng(seed)

    if punto_cambio is not None:
        if n_segmentos != 2:
            raise ValueError(
                "punto_cambio solo está soportado cuando n_segmentos == 2."
            )
        if not (0 < punto_cambio < N):
            raise ValueError(
                "punto_cambio debe estar estrictamente entre 0 y N."
            )
        longitudes = [punto_cambio, N - punto_cambio]

    else:
        if N < n_segmentos * min_sep or N > n_segmentos * max_sep:
            raise ValueError(
                "N debe estar entre n_segmentos * min_sep y "
                "n_segmentos * max_sep para poder acomodar todos los segmentos."
            )

        longitudes = []
        restante = N

        for i in range(n_segmentos - 1):
            seg_restantes = n_segmentos - i - 1
            minimo_factible = max(min_sep, restante - seg_restantes * max_sep)
            maximo_factible = min(max_sep, restante - seg_restantes * min_sep)
            L = rng.integers(minimo_factible, maximo_factible + 1)
            longitudes.append(L)
            restante -= L

        longitudes.append(restante)

    regimen_A = (1.0, -0.8)
    regimen_B = (-1.0, -0.8)

    regimen_actual = rng.choice([0, 1])

    parametros = []
    for _ in range(n_segmentos):
        if regimen_actual == 0:
            parametros.append(regimen_A)
            regimen_actual = 1
        else:
            parametros.append(regimen_B)
            regimen_actual = 0

    y = np.zeros(N)
    y[0] = rng.normal(A, sigma)
    y[1] = rng.normal(A, sigma)

    cps = []
    segmentos = []
    pos = 0

    for idx, (L, (phi1, phi2)) in enumerate(zip(longitudes, parametros)):
        inicio = pos
        fin = pos + L

        segmentos.append({
            "segmento": idx,
            "inicio": inicio,
            "fin": fin - 1,
            "longitud": L,
            "phi1": phi1,
            "phi2": phi2
        })

        if idx > 0:
            cps.append(inicio)

        t_inicio = max(inicio, 2)
        for t in range(t_inicio, fin):
            eps = rng.normal(0, sigma)
            y[t] = (phi1 * y[t - 1] + phi2 * y[t - 2]) + eps

        pos = fin

    return y, np.array(cps), segmentos