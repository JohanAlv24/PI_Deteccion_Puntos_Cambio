import numpy as np
from Algoritmo_Gaussiano.cpd import CPD
from Algoritmo_Empiricas.Empirical_CPD import EmpiricalCPD
from Series_Prueba.periodical_data import generar_series_pc, next_prob, serie_pc
from Series_Prueba.ARIMA import arima_serie
from Utils.detection import detect
from Utils.metrics_sup import metrics
from Series_Prueba.experimentos import samples_200_arma, samples_200_sin, ar2_noise
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def graficar_mapa_calor(start, end, CPD, espacio, title, path, show=False):
    print(f'Tiempo de ejecución {end-start} segundos')
        

    print(f'Mejor ventana método gaussiano: {CPD.window}')
    print(f'Mejor retardo método gaussiano: {CPD.t}')

    ws = sorted(set(k[0] for k in espacio.keys()))
    ts = sorted(set(k[1] for k in espacio.keys()))
    cost_matrix = np.full((len(ws), len(ts)), np.nan)

    for (w, t), cost in espacio.items():
        i = ws.index(w)
        j = ts.index(t)
        cost_matrix[i, j] = cost

    plt.figure(figsize=(10,6))

    plt.imshow(cost_matrix, aspect='auto', origin='lower')
    plt.colorbar(label="Costo")

    step_x = max(1, len(ts)//30)
    step_y = max(1, len(ws)//30)

    plt.xticks(range(0, len(ts), step_x), ts[::step_x], rotation=45)
    plt.yticks(range(0, len(ws), step_y), ws[::step_y])

    plt.xlabel("t")
    plt.ylabel("w")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    
def graficar_dispersion_costo(start, end, CPD, espacio, title, path):

    print(f'Tiempo de ejecución {end-start} segundos')
    print(f'Mejor ventana método empírico: {CPD.window}')

    ws_emp = np.array(sorted(espacio.keys()))
    costs_emp = np.array([espacio[w] for w in ws_emp])

    plt.figure(figsize=(10,6))

    plt.scatter(ws_emp, costs_emp, s=30)

    plt.xlabel("Tamaño de ventana (w)")
    plt.ylabel("Costo")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")

def cpd_serie_periodica(path, tran_mat, exp, pc_params, min_w1, n, length, sigma_amp, 
                        sigma_freq, sigma_fase, sigma_pend, sigma_ruido, gauss, 
                        min_w2=None, max_w=None, penal=True, lambda_p = 1e-3, thr=30, seed1=123, seed2=123, heuristic=False, max_iter=50, show=False):

    cps, cambios, clusters_cps = serie_pc(tran_mat, [exp]*len(tran_mat), pc_params, min_w1, n, seed1)

    cps_principales = {'S1': [j for j in cps for i in range(len(pc_params[0])) ]}

    cambios_cp = {'S1': cambios}

    amplitud_base, frecuencia_base, fase_base = pc_params[0]
    pendiente_base = 0

    datos_clustering, subgrupos = generar_series_pc(1, length, 1, 1, amplitud_base, 
                        frecuencia_base, pendiente_base, fase_base, sigma_amp, sigma_freq,
                        sigma_pend, sigma_ruido, cps_principales=cps_principales, cambios_cp=cambios_cp, aleatorio=False)
    
    serie_ruido = datos_clustering[0]
    T_ruido = len(serie_ruido)

    np.random.seed(seed2)
    if min_w2==None:
        min_w2 = 9
    if max_w==None:
        max_w = T_ruido//10

    if gauss:
        CPD_Periodica = CPD(serie_ruido, window = 0, t = 0, m = 3,
                            medias = True, sigma = 6, k_gauss = True)
        start = time.time()
        if penal:
            if heuristic:
                best_dist, best_cp = CPD_Periodica.heuristic_window_t(min_w = min_w2, max_w = max_w, penal = penal, lambda_p = lambda_p, max_iter=max_iter)
            else:
                best_dist, best_cp, espacio, f_cost, penalizaciones = CPD_Periodica.opt_window_t(min_w = min_w2, max_w = max_w, penal = penal, lambda_p = lambda_p)
        else:
            best_dist, best_cp, espacio = CPD_Periodica.opt_window_t(min_w = min_w2, max_w = max_w, penal = penal, lambda_p = lambda_p)
        end = time.time()
        if not heuristic:
            graficar_mapa_calor(start, end, CPD_Periodica, espacio, 'Mapa de calor costo serie periódica '+str(length), path, show=show)
        else:
            print(f'Tiempo de ejecución {end-start}')
            print(f'Mejor ventana: {CPD_Periodica.window} y mejor retardo: {CPD_Periodica.t}')
        if penal and not heuristic:
            graficar_mapa_calor(start, end, CPD_Periodica, f_cost, 'Mapa de calor solo costo serie periódica '+str(length), path+' solo coste')
            graficar_mapa_calor(start, end, CPD_Periodica, penalizaciones, 'Mapa de calor penalización serie periódica '+str(length), path+' solo penalización')
        w_b = CPD_Periodica.window
        metricas_cpd = metrics(cps, best_cp, thr, T_ruido)
        metricas_cpd[0]['w'] = w_b

        
    else:
        CPD_Periodica_emp = EmpiricalCPD(serie_ruido, window=30, sigma=6, k_gauss=True)

        start = time.time()
        if penal:
            best_dist_emp, best_cp_emp, espacio_emp, f_cost, penalizaciones = CPD_Periodica_emp.opt_window(
                min_w=min_w2,
                max_w=max_w,
                penal=penal,
                lambda_p=lambda_p
            )
        else:
            best_dist_emp, best_cp_emp, espacio_emp = CPD_Periodica_emp.opt_window(
                min_w=min_w2,
                max_w=max_w,
                penal=penal,
                lambda_p=lambda_p
            )

        end = time.time()

        graficar_dispersion_costo(start, end, CPD_Periodica_emp, espacio_emp, "Costo vs tamaño de ventana (método empírico)", path)
        if penal:
            graficar_dispersion_costo(start, end, CPD_Periodica_emp, f_cost, "Solo Costo vs tamaño de ventana (método empírico) "+str(length), path+' solo coste')
            graficar_dispersion_costo(start, end, CPD_Periodica_emp, penalizaciones, "Penalización vs tamaño de ventana (método empírico) "+str(length), path+' solo penalización')
        w_b = CPD_Periodica_emp.window
        metricas_cpd = metrics(cps, best_cp_emp, thr, T_ruido)
        metricas_cpd[0]['w'] = w_b
    if not heuristic:
        return metricas_cpd, espacio
    return metricas_cpd


def cpd_serie_arma(path, T, changes_arima, param_changes, p, q, thr=30, min_w=None, max_w=None, penal=True, lambda_p=1e-3, gauss=True, seed=123, heuristic=False):

    x_arima = arima_serie(
                T=T,
                change_points=changes_arima,
                param_changes=param_changes,
                p=p,
                q=q,
                seed=42)
    

    if min_w==None:
        min_w = 9
    if max_w==None:
        max_w = T//10

    plt.plot(x_arima)
    plt.title(f'Serie de tiempo ARMA p={p} y q={q}')
    for i in changes_arima:
        plt.axvline(i, color="k", linestyle="--", alpha=0.3)

    if gauss:
        CPD_arma = CPD(x_arima, window = 0, t = 0, m = 3,
                            medias = True, sigma = 6, k_gauss = True)
        start = time.time()
        
        if penal:
            if heuristic:
                distancias_arima, pc_detectados_arima = CPD_arma.heuristic_window_t(min_w = min_w, max_w = max_w, penal = penal, lambda_p = lambda_p)
            else:    
                distancias_arima, pc_detectados_arima, espacio, f_cost, penalizaciones = CPD_arma.opt_window_t(min_w = min_w, max_w = max_w, penal = penal, lambda_p = lambda_p)
        
        else:
            distancias_arima, pc_detectados_arima, espacio = CPD_arma.opt_window_t(min_w = min_w, max_w = max_w, penal = penal, lambda_p = lambda_p)
        end = time.time()
        if not heuristic:
            graficar_mapa_calor(start, end, CPD_arma, espacio, f'Mapa de calor costo serie ARMA p={p} y q={q}', path)
        else:
            print(f'Tiempo de ejecución {end-start}')
            print(f'Mejor ventana: {CPD_arma.window} y mejor retardo: {CPD_arma.t}')
        metricas_cpd = metrics(changes_arima, pc_detectados_arima, thr, T)
        w_b = CPD_arma.window
        metricas_cpd[0]['w'] = w_b

    else:

        CPD_arma_emp = EmpiricalCPD(x_arima, window=30, sigma=6, k_gauss=True)

        start = time.time()

        if penal:
            if heuristic:
                best_dist_emp, best_cp_emp = CPD_arma_emp.opt_window(
                    min_w=min_w,
                    max_w=max_w,
                    penal=penal,
                    lambda_p=lambda_p
                )
            else:
                best_dist_emp, best_cp_emp, espacio_emp, f_cost, penalizaciones = CPD_arma_emp.opt_window(
                    min_w=min_w,
                    max_w=max_w,
                    penal=penal,
                    lambda_p=lambda_p
                )
        else:
            best_dist_emp, best_cp_emp, espacio_emp = CPD_arma_emp.opt_window(
                min_w=min_w,
                max_w=max_w,
                penal=penal,
                lambda_p=lambda_p
            )

        end = time.time()

        graficar_dispersion_costo(start, end, CPD_arma_emp, espacio_emp, "Costo vs tamaño de ventana (método empírico)", path)

        
        w_b = CPD_arma_emp.window
        metricas_cpd = metrics(changes_arima, best_cp_emp, thr, T)
        metricas_cpd[0]['w'] = w_b

    return metricas_cpd

def boxplot_comp(M1, M2, columnas, title):
    k = M1.shape[1]

    df_list = []

    for i, col in enumerate(columnas[1:4]):
        df_list.append(pd.DataFrame({
            "Valor": M1[:, i+1],
            "Columna": col,
            "Matriz": "Gaussiana"
        }))
        
        df_list.append(pd.DataFrame({
            "Valor": M2[:, i+1],
            "Columna": col,
            "Matriz": "Empírica"
        }))

    df = pd.concat(df_list)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Columna", y="Valor", hue="Matriz")
    plt.title(title)
    plt.show()

    df_list = []
    col_may = [columnas[0]] + columnas[4:]
    for i, col in enumerate(col_may[0:1]):
        df_list.append(pd.DataFrame({
            "Valor": M1[:, i+1],
            "Columna": col,
            "Matriz": "Gaussiana"
        }))
        
        df_list.append(pd.DataFrame({
            "Valor": M2[:, i+1],
            "Columna": col,
            "Matriz": "Empírica"
        }))

    df = pd.concat(df_list)

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df, x="Columna", y="Valor", hue="Matriz")
    plt.title(title)
    plt.show()

def arma_exp(seed, penal, path, window=30, t=0, m=0, f_gauss=6,
            T=2000, phi=(0.3, 0.5), theta=(0.0, 0.0), random_phi=False,
            random_theta=False, min_seg=50, max_seg=150, base_mean=0.0,
            random_mean=True, mean_range=(-1.0, 1.0), base_std=0.5, 
            random_std=True, std_range=(0.2, 1.2), outlier_interval=200,
            outlier_scale=6.0, seed2=None):
    np.random.seed(seed)
    dataset, cps, outliers = ar2_noise(T, phi, theta, random_phi,
                                        random_theta, min_seg, max_seg, base_mean,
                                        random_mean, mean_range, base_std,
                                        random_std, std_range, outlier_interval,
                                        outlier_scale, seed2
                                        )

    start = time.time()
    PC_dataset1_emp = EmpiricalCPD(dataset)
    best_dist, pc_detectados_dataset1_emp, espacio, f_costos_emp, penalizaciones_emp = PC_dataset1_emp.opt_window(max_w=T//10, penal=penal, lambda_p = 1/(3*np.log(T)*T**0.5))
    end = time.time()
    graficar_dispersion_costo(start, end, PC_dataset1_emp, espacio, " Costo vs tamaño de ventana (método empírico) "+str(T), path+' empírica coste con penalización')
    graficar_dispersion_costo(start, end, PC_dataset1_emp, f_costos_emp, "Solo Costo vs tamaño de ventana (método empírico) "+str(T), path+' empírica solo coste')
    graficar_dispersion_costo(start, end, PC_dataset1_emp, penalizaciones_emp, "Penalización vs tamaño de ventana (método empírico) "+str(T), path+' empírica solo penalización')

    return metrics(cps, pc_detectados_dataset1_emp, 30, T)

if __name__ == "__main__":
    casos_base = True
    if casos_base:
        penal = True
        heuristic = True
        #SERIE PERIÓDICA
        tran_mat = np.array([[0, 1/3, 5/12, 1/4],
                        [1/5, 0, 2/5, 2/5],
                        [1/7, 2/7, 0, 4/7],
                        [1/4, 1/2, 1/4, 0]])
        
        pc_params = [[5, 0.05, 2.0], 
                [2, 0.2, 0.5],
                [7.0, 0.03, 0.5],
                [0.05, 0.008, 2.0]]

        np.random.seed(123)   

        sigma_amp = 0.1
        sigma_freq = 0.003
        sigma_fase = 0.05
        sigma_pend = 0.0
        sigma_ruido = 0.8

        exp=20
        min_w1=30
        n=40
        length = 2000

        lambda_p = 1/(3*np.log(length)*length**0.5)
        metricas_periodica_emp   = cpd_serie_periodica('Gráficas/Penalización/Dispersión_Periódica_Emp', tran_mat, exp, pc_params, min_w1, n, length, sigma_amp, 
                                                    sigma_freq, sigma_fase, sigma_pend, sigma_ruido, gauss=False, penal=penal, lambda_p=lambda_p)
        metricas_periodica_gauss, espacio = cpd_serie_periodica('Gráficas/Penalización/mapa_Periódica_gauss', tran_mat, exp, pc_params, min_w1, n, length, sigma_amp, 
                                                        sigma_freq, sigma_fase, sigma_pend, sigma_ruido, gauss=True, penal=penal, lambda_p=lambda_p, heuristic=heuristic, max_iter=100, show=True)
            
        print(f'Función de coste para w: {43} y t: {8} es: {espacio[(43, 8)]}')
        print(f'Función de coste para w: {12} y t: {1} es: {espacio[(12, 1)]}')
        print(f'Función de coste para w: {16} y t: {1} es: {espacio[(16, 1)]}')
        print(f'Función de coste para w: {52} y t: {3} es: {espacio[(52, 3)]}')

        #SERIE ARIMA 1
        lambda_p = 1/(3*np.log(2000)*2000*0.5)
        T1 = 2000
        p1 = 3
        q1 = 2
        changes_arima1 = [0, 120, 260, 410, 580, 750, 900, 1084, 1229, 1345, 1550, 1620, 1730, 1900]
        A = {
                'phi': [0.55, -0.30, 0.15],
                'theta': [0.60, -0.25],
                'sigma': 2.5,
                'c': 0.0
                }
        B = {
                'phi': [-0.35, 0.20, -0.10],
                'theta': [0.25, -0.10],
                'sigma': 0.3,
                'c': 0.0
                }
        C = {
                'phi': [0.75, -0.35, 0.20],
                'theta': [0.40, 0.25],
                'sigma': 1.0,
                'c': 0.0
                }
        param_changes = [A, B, A, B, C, A, C, B, C, A, B, C, A, C]

        metricas_arima_gauss = cpd_serie_arma('Gráficas/Penalización/mapa_arma_gauss',T1, changes_arima1, param_changes, p1, q1, penal=penal, gauss=True, lambda_p=lambda_p, heuristic=heuristic)
        metricas_arima_emp = cpd_serie_arma('Gráficas/Penalización/Dispersión_arma_Emp',T1, changes_arima1, param_changes, p1, q1, penal=penal, gauss=False, lambda_p=lambda_p)

            #SERIE AR(6)
        T2 = 2000
        p2 = 6
        q2 = 1

        changes_2 = [0, 90, 170, 260, 350, 450, 560, 690, 820, 960, 1100, 1260, 1430, 1600, 1780, 1950]

        A1 = {
                'phi': [0.78, -0.40, 0.25, -0.15, 0.08, -0.04],
                'theta': [0.15],
                'sigma': 1.8,
                'c': 0.0
            }

        B1 = {
                'phi': [0.60, -0.35, 0.20, -0.10, 0.05, -0.02],
                'theta': [0.10],
                'sigma': 0.5,
                'c': 0.0
            }

        C1 = {
                'phi': [0.85, -0.50, 0.30, -0.20, 0.12, -0.06],
                'theta': [0],
                'sigma': 3.2,
                'c': 0.0
            }

        param_changes_2 = [C1, A1, B1, C1, A1, B1, C1, A1, C1, A1, B1, C1, A1, B1, C1, A1]

        metricas_ar6_gauss = cpd_serie_arma('Gráficas/Penalización/mapa_ar6_gauss',T2, changes_2, param_changes_2, p2, q2, penal=penal, gauss=True, lambda_p=lambda_p, heuristic=heuristic)
        metricas_ar6_emp = cpd_serie_arma('Gráficas/Penalización/Dispersión_ar6_Emp',T2, changes_2, param_changes_2, p2, q2, penal=penal, gauss=False, lambda_p=lambda_p)    

        #SERIE ARIMA 2
        T3 = 2100
        p3 = 3
        q3 = 3

        changes_3 = [0, 80, 150, 230, 320, 410, 500, 600, 720, 860, 1000, 1150, 1310, 1470, 1640, 1820, 1980]

        A2 = {
                'phi': [0.40, -0.25, 0.10],
                'theta': [0.60, -0.40, 0.25],
                'sigma': 2.0,
                'c': 0.0
            }

        B2 = {
                'phi': [-0.35, 0.20, -0.10],
                'theta': [0.70, -0.60, 0.30],
                'sigma': 0.4,
                'c': 0.0
            }

        C2 = {
                'phi': [0.50, -0.30, 0.15],
                'theta': [0.55, 0.25, -0.20],
                'sigma': 1.0,
                'c': 0.0
            }

        param_changes_3 = [A2, B2, C2, B2, A2, C2, B2, A2, C2, A2, B2, C2, A2, C2, B2, A2, C2]

        metricas_arima2_gauss = cpd_serie_arma('Gráficas/Penalización/mapa_arima2_gauss',T3, changes_3, param_changes_3, p3, q3, penal=penal, gauss=True, lambda_p=lambda_p, heuristic=heuristic)
        metricas_arima2_emp = cpd_serie_arma('Gráficas/Penalización/Dispersión_arima_Emp',T3, changes_3, param_changes_3, p3, q3, penal=penal, gauss=False, lambda_p=lambda_p)    


        #RESUMEN MÉTRICAS
        print('MÉTRICAS CDP SERIE PERIÓDICA MÉTODO GAUSSIANO')
        print(metricas_periodica_gauss)
        print()
        print('MÉTRICAS CDP SERIE PERIÓDICA MÉTODO EMPÍRICA')
        print(metricas_periodica_emp)
        print()

        print('MÉTRICAS CDP SERIE ARMA MÉTODO GAUSSIANO')
        print(metricas_arima_gauss)
        print()
        print('MÉTRICAS CDP SERIE ARMA MÉTODO EMPÍRICA')
        print(metricas_arima_emp)
        print()

        print('MÉTRICAS CDP SERIE AR6 MÉTODO GAUSSIANO')
        print(metricas_ar6_gauss)
        print()
        print('MÉTRICAS CDP SERIE AR6 MÉTODO EMPÍRICA')
        print(metricas_ar6_emp)
        print()

        print('MÉTRICAS CDP SERIE ARMA 2 MÉTODO GAUSSIANO')
        print(metricas_arima2_gauss)
        print()
        print('MÉTRICAS CDP SERIE ARMA 2 MÉTODO EMPÍRICA')
        print(metricas_arima2_emp)
        print()

        nombres = ['SERIE PERIÓDICA MÉTODO GAUSSIANO', 'SERIE PERIÓDICA MÉTODO EMPÍRICA', 'SERIE ARMA MÉTODO GAUSSIANO', 'SERIE ARMA MÉTODO EMPÍRICA', 
        'SERIE AR6 MÉTODO GAUSSIANO', 'SERIE AR6 MÉTODO EMPÍRICA', 'SERIE ARMA 2 MÉTODO GAUSSIANO', 'SERIE ARMA 2 MÉTODO EMPÍRICA']
            

        diccionarios = [metricas_periodica_gauss[0], metricas_periodica_emp[0], metricas_arima_gauss[0], metricas_arima_emp[0], metricas_ar6_gauss[0], 
                            metricas_ar6_emp[0], metricas_arima2_gauss[0], metricas_arima2_emp[0]]
            
        df = pd.DataFrame(diccionarios)
        df.insert(0, 'Metodo', nombres)
        df.to_excel('Tablas_Métricas/Métricas_heurística.xlsx', index=False)
    

    experimentos = False
    if experimentos:
        
        print("EXPERIMENTO AR(2) CON MEDIA Y DISPERSIÓN FLUCTUANTES")
        met_gauss_ar2_1, met_emp_ar2_1 = samples_200_arma(seed=1234, penal=True, lambda_p=1e-3, thr_dist=30,
                                                        base_mean=0.3, std_range=(0.3, 1.2), seed2=None)
        print()

        print("EXPERIMENTO AR(2) CON REZAGOS FLUCTUANTES")
        met_gauss_ar2_2, met_emp_ar2_2 = samples_200_arma(seed=1234, penal=True, lambda_p=1e-3, thr_dist=30,
                                                        random_phi=True, min_seg=50, max_seg=150,  base_mean=0.5, 
                                                        random_mean=False, base_std=0.5, random_std=True, std_range=(0.3, 1.2))
        print()

        print("EXPERIMENTO ARMA CON REZAGOS FLUCTUANTES")
        met_gauss_arma, met_emp_arma = samples_200_arma(seed=1234, penal=True, lambda_p=1e-3, thr_dist=30,
                                                        random_phi=True, random_theta=True, min_seg=50, max_seg=150,  base_mean=0.5, 
                                                        random_mean=False, base_std=0.5, random_std=True, std_range=(0.3, 1.2))
        print()

        print("EXPERIMENTO SERIE PERIÓDICA")
        tran_mat = np.array([[0, 1/3, 5/12, 1/4],
                        [1/5, 0, 2/5, 2/5],
                        [1/7, 2/7, 0, 4/7],
                        [1/4, 1/2, 1/4, 0]])
        
        pc_params = [[5, 0.05, 2.0], 
                    [2, 0.2, 0.5],
                    [7.0, 0.03, 0.5],
                    [0.05, 0.008, 2.0]]
        
        met_gauss_sin, met_emp_sin = samples_200_sin(tran_mat, 20, pc_params, min_w=60, n=40, penal=True, lambda_p=1e-3, thr_dist=30)

        np.savez(
                    "metricas_opt.npz",
                    M1=met_gauss_ar2_1,
                    M2=met_emp_ar2_1,
                    M3=met_gauss_ar2_2,
                    M4=met_emp_ar2_2,
                    M5=met_gauss_arma,
                    M6=met_emp_arma,
                    M7=met_gauss_sin,
                    M8=met_emp_sin
                )
    
    casos_orden = False
    if casos_orden:
        penal = True
        #SERIE PERIÓDICA
        tran_mat = np.array([[0, 1/3, 5/12, 1/4],
                        [1/5, 0, 2/5, 2/5],
                        [1/7, 2/7, 0, 4/7],
                        [1/4, 1/2, 1/4, 0]])
        
        pc_params = [[5, 0.05, 2.0], 
                [2, 0.2, 0.5],
                [7.0, 0.03, 0.5],
                [0.05, 0.008, 2.0]]

        np.random.seed(123)   

        sigma_amp = 0.1
        sigma_freq = 0.003
        sigma_fase = 0.05
        sigma_pend = 0.0
        sigma_ruido = 0.8
        exp=20

        
        #Tamaño 2000
        min_w1=30
        n=400
        length = 20000

        #metricas_periodica100_emp   = cpd_serie_periodica('Gráficas/Orden/Dispersión_Periódica20,000_Emp', tran_mat, exp, pc_params, min_w1, n, length, sigma_amp, 
        #                                            sigma_freq, sigma_fase, sigma_pend, sigma_ruido, gauss=False, penal=penal, lambda_p=1/(3*np.log(length)*length**0.5))
        
        #print(metricas_periodica100_emp)
        
        arma_exp(seed=1234, penal=True, path='Gráficas/Orden/ARMA/200', T=200,
                random_phi=True, random_theta=True, min_seg=30, max_seg=50,  base_mean=0.5, 
                random_mean=False, base_std=0.5, random_std=True, std_range=(0.3, 1.2))
        arma_exp(seed=1234, penal=True, path='Gráficas/Orden/ARMA/2,000', T=2000,
                random_phi=True, random_theta=True, min_seg=30, max_seg=50,  base_mean=0.5, 
                random_mean=False, base_std=0.5, random_std=True, std_range=(0.3, 1.2))
        
        arma_exp(seed=1234, penal=True, path='Gráficas/Orden/ARMA/20,000', T=20000,
                random_phi=True, random_theta=True, min_seg=30, max_seg=50,  base_mean=0.5, 
                random_mean=False, base_std=0.5, random_std=True, std_range=(0.3, 1.2))
        




    #data = np.load("metricas_opt.npz")

    #M1_opt = data["M1"]
    #M2_opt = data["M2"]
    #M3_opt = data["M3"]
    #M4_opt = data["M4"]
    #M5_opt = data["M5"]
    #M6_opt = data["M6"]
    #M7_opt = data["M7"]
    #M8_opt = data["M8"]

    #columnas = ['Mean Location Error', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'Falsos Positivos', 'Falsos Negativos', 'Verdaderos Positivos']

    #boxplot_comp(M1_opt, M2_opt, columnas, "Comparación métricas AR variando media y dispersión")
    #boxplot_comp(M3_opt, M4_opt, columnas, "Comparación métricas AR variando rezagos")
    #boxplot_comp(M5_opt, M6_opt, columnas, "Comparación métricas ARMA")
    #boxplot_comp(M7_opt, M8_opt, columnas, "Comparación métricas Serie Periódica")

    


        
    
