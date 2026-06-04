import numpy as np
import roerich
from roerich.change_point import ChangePointDetectionClassifier
from Algoritmo_Gaussiano.cpd import CPD
from Algoritmo_Empiricas.Empirical_CPD import EmpiricalCPD
from Utils.detection import detect
from Utils.metrics_sup import metrics
from Utils.tools import (
    best_params_sh,
    print_toolbox_header,
    print_toolbox_section,
    print_toolbox_item,
    print_metrics_block,
    print_toolbox_success,
    print_toolbox_warning
)

from Series_Prueba.periodical_data import generar_series_pc, serie_pc



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
    phi=(0.3, 0.5),
    theta=(0.0, 0.0),
    random_phi=False,
    random_theta=False,
    #phi_range=(-0.9, 0.9),
    #theta_range=(-0.9, 0.9),
    min_seg=50,
    max_seg=200,
    base_mean=0.0,
    random_mean=True,
    mean_range=(-1.0, 1.0),
    base_std=0.5,
    random_std=True,
    std_range=(0.2, 1.2),
    outlier_interval=200,
    outlier_scale=6.0,
    seed=None
):
    if seed is not None:
        np.random.seed(seed)

    x = np.zeros(T)
    eps = np.zeros(T)

    cps = []
    t = 0

    phi_t = np.array(phi)
    theta_t = np.array(theta)

    while t <= T - min_seg:
        seg_len = np.random.randint(min_seg, max_seg + 1)
        end = min(t + seg_len, T)

        if random_mean:
            x_mu = np.random.uniform(*mean_range)
            mu = x_mu + np.exp(-abs(x_mu))
        else:
            mu = base_mean

        if random_std:
            sigma = np.random.uniform(*std_range)
        else:
            sigma = base_std

        if random_phi:
            phi_t = sample_stationary_ar2()

        if random_theta:
            theta_t = sample_invertible_ma2()

        eps[t:end] = np.random.normal(mu, sigma, end - t)

        if t not in cps and t != 0:
            cps.append(t)
        if end not in cps and end != T:
            cps.append(end)

        for i in range(max(t, 2), end):
            x[i] = (
                phi_t[0] * x[i - 1]
                + phi_t[1] * x[i - 2]
                + eps[i]
                + theta_t[0] * eps[i - 1]
                + theta_t[1] * eps[i - 2]
            )

        t = end

    outlier_idx = np.arange(outlier_interval, T, outlier_interval)

    eps[outlier_idx] += np.random.normal(
        0.0,
        outlier_scale * np.std(eps),
        size=len(outlier_idx)
    )

    x[outlier_idx] += eps[outlier_idx]

    return x, sorted(cps), outlier_idx


def samples_200_arma(
    seed,
    N=200,
    thr_dist=30,
    T=2000,
    phi=(0.3, 0.5),
    theta=(0.0, 0.0),
    random_phi=False,
    random_theta=False,
    min_seg=50,
    max_seg=150,
    base_mean=0.0,
    random_mean=True,
    mean_range=(-1.0, 1.0),
    base_std=0.5,
    random_std=True,
    std_range=(0.2, 1.2),
    outlier_interval=200,
    outlier_scale=6.0,
    s_thresh=0,
    seed2=None,
    tail=False,
    tail_list=None,
    given=True,
    geom=True,
    visualization=None
):

    if tail_list is None:
        tail_list = [N]

    if visualization is None:
        visualization = {}

    summary_enabled = visualization.get("summary", False)
    raw_enabled = visualization.get("raw_distance", False)
    smooth_enabled = visualization.get("smooth_distance", False)
    cost_enabled = visualization.get("cost", False)
    slope_enabled = visualization.get("slope", False)

    print_toolbox_header("Serie sintética ARMA")
    print_toolbox_item("Experimentos totales", N)


    np.random.seed(seed)

    metricas_gauss = []
    metricas_emp = []

    ejecutados = 0

    for i in range(N):

        seed_actual = None if seed2 is None else seed2 + i

        dataset1, cps_ar, outliers_ar = ar2_noise(
            T,
            phi,
            theta,
            random_phi,
            random_theta,
            min_seg,
            max_seg,
            base_mean,
            random_mean,
            mean_range,
            base_std,
            random_std,
            std_range,
            outlier_interval,
            outlier_scale,
            seed_actual
        )

        ejecutar = (not tail) or ((i + 1) in tail_list)

        if not ejecutar:
            continue

        ejecutados += 1

        print_toolbox_header(f"Experimento {i + 1}/{N}")

        print_toolbox_item("Longitud de serie", T)

        print_toolbox_item(
            "Semilla del experimento",
            seed_actual if seed_actual is not None else "Aleatoria"
        )

        print_toolbox_item(
            "Puntos de cambio reales",
            len(cps_ar)
        )


        met_dataset1_gauss = best_params_sh(
            dataset1,
            cps_ar,
            s_thresh=s_thresh,
            penal=True,
            lambda_p=0,
            given=given,
            geom=geom,
            visualization={
                "summary": summary_enabled,
                "raw_distance": raw_enabled,
                "smooth_distance": smooth_enabled,
                "cost": cost_enabled,
                "slope": slope_enabled
            }
        )

        met_dataset1_emp = best_params_sh(
            dataset1,
            cps_ar,
            s_thresh=s_thresh,
            penal=True,
            lambda_p=0,
            gauss=False,
            given=given,
            geom=geom,
            visualization={
                "summary": summary_enabled,
                "raw_distance": raw_enabled,
                "smooth_distance": smooth_enabled,
                "cost": cost_enabled,
                "slope": slope_enabled
            }
        )

        print_toolbox_section("Resumen de métricas")

        print_metrics_block(
            "Gaussiano",
            met_dataset1_gauss[0]
        )

        print_metrics_block(
            "Empírico",
            met_dataset1_emp[0]
        )

        metricas_gauss.append(
            list(met_dataset1_gauss[0].values())
        )

        metricas_emp.append(
            list(met_dataset1_emp[0].values())
        )

        print_toolbox_success(
            f"Experimento {i + 1}/{N} finalizado"
        )

    print_toolbox_header("Resumen global")

    print_toolbox_item(
        "Experimentos ejecutados",
        ejecutados
    )

    print_toolbox_item(
        "Experimentos solicitados",
        N
    )

    return metricas_gauss, metricas_emp

def samples_200_sin(tran_mat, exp, pc_params, min_w, n, penal, lambda_p, N=200, w=30, t=0, m=0, seed=None, thr_dist=30,
                     sigma_amp = 0.1, sigma_freq = 0.003, sigma_fase = 0.05, sigma_pend = 0.0,
                     sigma_ruido = 0.8):
    if seed is not None:
        np.random.seed(seed)

    print_toolbox_header("Serie sintética periódica")
    print_toolbox_item("Experimentos totales", N)

    amplitud_base, frecuencia_base, fase_base = pc_params[0]
    pendiente_base = 0

    metricas_gauss = []
    metricas_emp = []

    for i in range(N):
        print_toolbox_header(f"Experimento {i + 1}/{N}")

        seed_actual = None if seed is None else seed + i
        cps, cambios, clusters_cps = serie_pc(tran_mat, [exp] * 4, pc_params, min_w, n, seed=seed_actual)

        cps_principales = {'S1': [j for j in cps for i in range(3)]}
        cambios_cp = {'S1': cambios}
        datos_clustering, subgrupos = generar_series_pc(
            1, cps[-1] + min_w, 1, 1, amplitud_base,
            frecuencia_base, pendiente_base, fase_base, sigma_amp, sigma_freq,
            sigma_pend, sigma_ruido, cps_principales=cps_principales, cambios_cp=cambios_cp, aleatorio=False, graficar=False
        )

        serie_ruido = datos_clustering[0]
        T = len(serie_ruido)

        PC_ruido = CPD(serie_ruido, window=w, t=t, m=m, medias=True, smooth=True)
        distancias, pc_detectados_gauss, espacio = PC_ruido.opt_window_t(max_w=T // 10, penal=penal, lambda_p=lambda_p)
        met_gauss, values_gauss = metrics(cps, pc_detectados_gauss, thr_dist, T)
        metricas_gauss.append(list(met_gauss.values()))

        PC_ruido_emp = EmpiricalCPD(serie_ruido)
        best_dist, pc_detectados_emp, espacio = PC_ruido_emp.opt_window(max_w=T // 10, penal=penal, lambda_p=lambda_p)
        met_emp, values_emp = metrics(cps, pc_detectados_emp, thr_dist, T)
        metricas_emp.append(list(met_emp.values()))

        print_toolbox_section("Resultados")
        print_metrics_block("Gaussiano", met_gauss)
        print_metrics_block("Empírico", met_emp)
        print_toolbox_success(f"Experimento {i + 1}/{N} finalizado")

    print_toolbox_header("Resumen global")
    print_toolbox_item("Experimentos ejecutados", N)

    return metricas_gauss, metricas_emp