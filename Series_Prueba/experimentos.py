import numpy as np
from Algoritmo_Gaussiano.cpd import CPD
from Algoritmo_Empiricas.Empirical_CPD import EmpiricalCPD
from Utils.detection import detect
from Utils.metrics_sup import metrics
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
                phi_t[0] * x[i-1]
                + phi_t[1] * x[i-2]
                + eps[i]
                + theta_t[0] * eps[i-1]
                + theta_t[1] * eps[i-2]
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


def samples_200_arma(seed, penal, lambda_p, N=200, window=30, t=0, m=0, f_gauss=6, thr_dist=30,
                    T=2000, phi=(0.3, 0.5), theta=(0.0, 0.0), random_phi=False,
                    random_theta=False, min_seg=50, max_seg=150, base_mean=0.0,
                    random_mean=True, mean_range=(-1.0, 1.0), base_std=0.5, 
                    random_std=True, std_range=(0.2, 1.2), outlier_interval=200,
                    outlier_scale=6.0, seed2=None):
    
    np.random.seed(seed)
    metricas_gauss = []
    metricas_emp = []
    for i in range(N):
        print(i)
        dataset1, cps_ar2, outliers_ar2 = ar2_noise(T, phi, theta, random_phi,
                                                        random_theta, min_seg, max_seg, base_mean,
                                                        random_mean, mean_range, base_std,
                                                        random_std, std_range, outlier_interval,
                                                        outlier_scale, seed2
                                                    )
        T = len(dataset1)
        PC_dataset1 = CPD(dataset1, window, t, m, True, f_gauss, True)
        distancias, pc_detectados_dataset1, espacio =  PC_dataset1.opt_window_t(max_w=T//10, penal=penal, lambda_p=lambda_p)
        met_dataset1, values_dataset1 = metrics(cps_ar2, pc_detectados_dataset1, thr_dist, T)
        metricas_gauss.append(list(met_dataset1.values()))

        PC_dataset1_emp = EmpiricalCPD(dataset1)
        best_dist, pc_detectados_dataset1_emp, espacio = PC_dataset1_emp.opt_window(max_w=T//10, penal=penal, lambda_p=lambda_p)
        met_dataset1_emp, values_dataset1_emp = metrics(cps_ar2, pc_detectados_dataset1_emp, thr_dist, T)
        metricas_emp.append(list(met_dataset1_emp.values()))
        
    return metricas_gauss, metricas_emp

def samples_200_sin(tran_mat, exp, pc_params, min_w, n, penal, lambda_p, N=200, w=30, t=0, m=0, seed=None, thr_dist=30, 
                     sigma_amp = 0.1, sigma_freq = 0.003, sigma_fase = 0.05, sigma_pend = 0.0, 
                     sigma_ruido = 0.8):
    if seed!=None:
        np.random.seed(seed)
    
    amplitud_base, frecuencia_base, fase_base = pc_params[0]
    pendiente_base = 0

    metricas_gauss = []
    metricas_emp = []

    for i in range(N):
        print(i)
        cps, cambios, clusters_cps = serie_pc(tran_mat, [exp]*4, pc_params, min_w, n, seed=None)

        cps_principales = {'S1': [j for j in cps for i in range(3) ]}
        cambios_cp = {'S1': cambios}
        datos_clustering, subgrupos = generar_series_pc(1, cps[-1]+min_w, 1, 1, amplitud_base, 
                             frecuencia_base, pendiente_base, fase_base, sigma_amp, sigma_freq,
                             sigma_pend, sigma_ruido, cps_principales=cps_principales, cambios_cp=cambios_cp, aleatorio=False, graficar=False)
    
        serie_ruido = datos_clustering[0]
        T = len(serie_ruido)

        PC_ruido = CPD(serie_ruido, w, t, m, True, 6, True)
        distancias, pc_detectados_gauss, espacio =  PC_ruido.opt_window_t(max_w=T//10, penal=penal, lambda_p=lambda_p)
        met_gauss, values_gauss = metrics(cps, pc_detectados_gauss, thr_dist, T)
        metricas_gauss.append(list(met_gauss.values()))
        
        PC_ruido_emp = EmpiricalCPD(serie_ruido)
        best_dist, pc_detectados_emp, espacio = PC_ruido_emp.opt_window(max_w=T//10, penal=penal, lambda_p=lambda_p)
        met_emp, values_emp = metrics(cps, pc_detectados_emp, thr_dist, T)
        metricas_emp.append(list(met_emp.values()))

    return metricas_gauss, metricas_emp
        
