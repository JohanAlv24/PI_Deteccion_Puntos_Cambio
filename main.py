import numpy as np
import time
from Algoritmo_Gaussiano.cpd import CPD
from Algoritmo_Empiricas.Empirical_CPD import EmpiricalCPD
from Series_Prueba.periodical_data import generar_series_pc, next_prob, serie_pc
from Series_Prueba.ARIMA import arima_serie
from Utils.detection import detect
from Utils.metrics_sup import metrics
from Series_Prueba.experimentos import samples_200_arma, samples_200_sin, ar2_noise
from matplotlib.ticker import MultipleLocator
import roerich
from roerich.change_point import ChangePointDetectionClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from Algoritmo_KLIEP.kliep_cpd import KLIEP_CPD
from Algoritmo_klcpd.klcpd import detect_changepoints as klcpd_detect

from Utils.tools import (
    best_params_sh)

import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns



def wilcoxon_metrica(arr_1,
                     arr_2,
                     minimizar=False,
                     alpha=0.05):
    """
    Realiza un test de Wilcoxon pareado entre dos arreglos.

    Parámetros
    ----------
    arr_1 : array-like
        Resultados del método 1.

    arr_2 : array-like
        Resultados del método 2.

    minimizar : bool, default=False
        - False -> métricas donde mayor es mejor.
        - True  -> métricas donde menor es mejor.

    alpha : float, default=0.05
        Nivel de significancia.

    Retorna
    -------
    dict
        Resultados estadísticos del test.
    """

    x = np.asarray(arr_1, dtype=float)
    y = np.asarray(arr_2, dtype=float)

    # Eliminación de NaN
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]

    # Wilcoxon pareado
    stat, p_value = wilcoxon(x, y)

    media_x = np.mean(x)
    media_y = np.mean(y)

    # Decisión estadística
    diferencia = p_value < alpha

    if diferencia:

        if minimizar:
            mejor = "Método 1" if media_x < media_y else "Método 2"
        else:
            mejor = "Método 1" if media_x > media_y else "Método 2"

    else:
        mejor = "Sin diferencia"

    return {
        "media_1": media_x,
        "media_2": media_y,
        "estadistico": stat,
        "p_value": p_value,
        "diferencia": diferencia,
        "mejor": mejor
    }


def comparar_metricas(metricas_metodo_1,
                      metricas_metodo_2,
                      nombre_metodo_1="Gaussiano",
                      nombre_metodo_2="Empírico",
                      alpha=0.05,
                      mostrar_tabla=True):
    """
    Compara múltiples métricas usando Wilcoxon pareado.

    Orden esperado:
    ----------------
    [0] Error Medio de Localización
    [1] Precisión
    [2] Exactitud
    [3] Medida F1
    """

    nombres_metricas = [
        "Error Medio de Localización",
        "Precisión",
        "Exactitud",
        "Medida F1"
    ]

    # True -> minimizar
    # False -> maximizar
    criterios = [
        True,
        False,
        False,
        False
    ]

    resultados = []

    for nombre, minimizar, arr1, arr2 in zip(
        nombres_metricas,
        criterios,
        metricas_metodo_1,
        metricas_metodo_2
    ):

        r = wilcoxon_metrica(
            arr1,
            arr2,
            minimizar=minimizar,
            alpha=alpha
        )

        if r["mejor"] == "Método 1":
            mejor = nombre_metodo_1
        elif r["mejor"] == "Método 2":
            mejor = nombre_metodo_2
        else:
            mejor = "Sin diferencia"

        resultados.append({
            "Métrica": nombre,
            f"Promedio {nombre_metodo_1}": r["media_1"],
            f"Promedio {nombre_metodo_2}": r["media_2"],
            "p-valor": r["p_value"],
            "Diferencia estadística": (
                "Sí" if r["diferencia"] else "No"
            ),
            "Mejor método": mejor
        })

    tabla = pd.DataFrame(resultados)

    if mostrar_tabla:

        print("\n" + "=" * 110)
        print("COMPARACIÓN ESTADÍSTICA ENTRE MÉTODOS (WILCOXON PAREADO)")
        print("=" * 110)

        print(tabulate(
            tabla,
            headers="keys",
            tablefmt="fancy_grid",
            showindex=False,
            floatfmt=".6f"
        ))

        print("\n")
        print(f"Nivel de significancia utilizado: alpha = {alpha}")

    return tabla

def boxplot_comp(M1, M2, columnas, title, path, usar_violin=False):
    sns.set(style="whitegrid", context="talk")

    def construir_df(cols_idx, cols_names):
        df_list = []
        for i, col in zip(cols_idx, cols_names):
            df_list.append(pd.DataFrame({
                "Valor": M1[:, i],
                "Columna": col,
                "Matriz": "Gaussiana"
            }))
            df_list.append(pd.DataFrame({
                "Valor": M2[:, i],
                "Columna": col,
                "Matriz": "Empírica"
            }))
        return pd.concat(df_list, ignore_index=True)

    def graficar(df, filename_suffix, y_step):
        plt.figure(figsize=(10, 6))

        if usar_violin:
            sns.violinplot(
                data=df, x="Columna", y="Valor", hue="Matriz",
                split=True, inner="box", palette="pastel"
            )
        else:
            sns.boxplot(
                data=df, x="Columna", y="Valor", hue="Matriz",
                palette="pastel", width=0.5
            )

            sns.stripplot(
                data=df, x="Columna", y="Valor", hue="Matriz",
                dodge=True, jitter=True, alpha=0.5,
                palette=["black", "gray"], size=3
            )

            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[:2], labels[:2], title="Matriz")

        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(y_step))

        plt.title(title)
        plt.tight_layout()
        plt.savefig(path + filename_suffix, dpi=300, bbox_inches="tight")
        plt.show()

    idx1 = list(range(1, 4))
    cols1 = columnas[1:4]
    df1 = construir_df(idx1, cols1)
    graficar(df1, ' 1', y_step=0.05)

    idx2 = [0]
    cols2 = [columnas[0]]
    df2 = construir_df(idx2, cols2)
    graficar(df2, ' 2', y_step=1)
'''
Función para probar los métodos de optimización en una serie de tiempo ARMA aleatoria (los puntos de cambio son aleatorios). 
Solo se utiliza el método de empíricas pues la idea de esta función es observar el comportamiento de la función de coste y la penalización
para distintos tamaños de la serie de tiempo
'''
def arma_exp(seed, penal, path, window=30, t=0, m=0, f_gauss=6,
            T=2000, phi=(0.3, 0.5), theta=(0.0, 0.0), random_phi=False,
            random_theta=False, min_seg=50, max_seg=150, base_mean=0.0,
            random_mean=True, mean_range=(-1.0, 1.0), base_std=0.5, 
            random_std=True, std_range=(0.2, 1.2), outlier_interval=200,
            outlier_scale=6.0, seed2=None, lambda_p=0, s_thresh=0):
    np.random.seed(seed)
    dataset, cps, outliers = ar2_noise(T, phi, theta, random_phi,
                                        random_theta, min_seg, max_seg, base_mean,
                                        random_mean, mean_range, base_std,
                                        random_std, std_range, outlier_interval,
                                        outlier_scale, seed2
                                        )

    start = time.time()
    plt.figure(figsize=(10,6))
    plt.plot(dataset)
    plt.show()
    print(len(cps))
    CPD_dataset1 = CPD(dataset)
    
    best_dist, pc_detectados_dataset1, f_costos, penalizaciones = CPD_dataset1.opt_window_t(max_w=T//10, penal=penal, lambda_p = lambda_p, join=False)
    end = time.time()
    '''if lambda_p != 0:
        graficar_dispersion_costo(start, end, CPD_dataset1, espacio, " Costo vs tamaño de ventana (método empírico) "+str(T), path+' empírica coste con penalización')'''

    graficar_mapa_calor(start, end, CPD_dataset1, f_costos, "Solo Costo vs tamaño de ventana "+str(T), path+' gaussiana solo coste')
    graficar_mapa_calor(start, end, CPD_dataset1, penalizaciones, "Penalización vs tamaño de ventana "+str(T), path+' gaussiana solo penalización')

    betha = CPD_dataset1.slope_heuristic_regression(s_thresh, f_costos, penalizaciones, plot=True, path='Gráficas/Heuristic_Slope/Visualización_Heuristic_Slope')

    return betha, f_costos, penalizaciones, metrics(cps, pc_detectados_dataset1, 30, T)


def generar_resumen(data):
    M1_opt = data["M1"]
    M2_opt = data["M2"]
    M3_opt = data["M3"]
    M4_opt = data["M4"]
    columnas = ['Mean Location Error', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'Falsos Positivos', 'Falsos Negativos', 'Verdaderos Positivos']
    

    print("EXPERIMENTO AR(2) CON MEDIA Y DISPERSIÓN FLUCTUANTES")
    print(f'Media F1 Gaussiana: {np.mean(M1_opt[:, 3])}')
    print(f'Media F1 Empírica: {np.mean(M2_opt[:, 3])}')

    met_gauss1 = np.array([M1_opt[:, 0], M1_opt[:, 1], M1_opt[:, 2], M1_opt[:, 3]])

    met_emp1 = np.array([M2_opt[:, 0], M2_opt[:, 1], M2_opt[:, 2], M2_opt[:, 3]])

    tabla_resultados1 = comparar_metricas(
                                            met_gauss1,
                                            met_emp1,
                                            nombre_metodo_1="Gaussiano",
                                            nombre_metodo_2="Empírico",
                                            alpha=0.05)
    boxplot_comp(M1_opt, M2_opt, columnas, "Comparación métricas AR variando media y dispersión", 'Gráficas/Umbral Suavizado/Comparación_AR_Media_', usar_violin=True)

    
    print("EXPERIMENTO AR(2) CON REZAGOS FLUCTUANTES")
    print(f'Media F1 Gaussiana: {np.mean(M3_opt[:, 3])}')
    print(f'Media F1 Empírica: {np.mean(M4_opt[:, 3])}')

    met_gauss2 = np.array([M3_opt[:, 0], M3_opt[:, 1], M3_opt[:, 2], M3_opt[:, 3]])
    met_emp2 = np.array([M4_opt[:, 0], M4_opt[:, 1], M4_opt[:, 2], M4_opt[:, 3]])
    tabla_resultados2 = comparar_metricas(
                                            met_gauss2,
                                            met_emp2,
                                            nombre_metodo_1="Gaussiano",
                                            nombre_metodo_2="Empírico",
                                            alpha=0.05)

    boxplot_comp(M3_opt, M4_opt, columnas, "Comparación métricas AR variando rezagos", 'Gráficas/Umbral Suavizado/Comparación_AR_Rezagos_', usar_violin=True)

def generar_informe(data):
    M1_opt = data["M1"]
    M2_opt = data["M2"]
    M3_opt = data["M3"]
    M4_opt = data["M4"]

    columnas = ['Mean Location Error', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'Falsos Positivos', 'Falsos Negativos', 'Verdaderos Positivos']
    

    ids = ["Gauss", "Empírico"]

    fila1 = [ids[0]] + list(np.mean(np.array(M1_opt), axis=0))
    fila2 = [ids[1]] + list(np.mean(np.array(M2_opt), axis=0))

    df_ar2_1 = pd.DataFrame([fila1, fila2], columns=['Método']+columnas)

    fila1 = [ids[0]] + list(np.mean(np.array(M3_opt), axis=0))
    fila2 = [ids[1]] + list(np.mean(np.array(M4_opt), axis=0))

    df_ar2_2 = pd.DataFrame([fila1, fila2], columns=['Método']+columnas)

    df_ar2_1.to_excel("Resultados/Resumen_AR_Media.xlsx", index=False)
    df_ar2_2.to_excel("Resultados/Resumen_AR_Rezagos.xlsx", index=False)

    df1 = pd.DataFrame(M1_opt, columns=columnas)
    df2 = pd.DataFrame(M2_opt, columns=columnas)
    df3 = pd.DataFrame(M3_opt, columns=columnas)
    df4 = pd.DataFrame(M4_opt, columns=columnas)
    
    # Exportar a un solo archivo Excel con múltiples hojas
    with pd.ExcelWriter("Resultados/metricas_opt2_suavizado.xlsx") as writer:
        df1.to_excel(writer, sheet_name="M1", index=False)
        df2.to_excel(writer, sheet_name="M2", index=False)
        df3.to_excel(writer, sheet_name="M3", index=False)
        df4.to_excel(writer, sheet_name="M4", index=False)



if __name__ == "__main__":
    
    
    visualization = {
                        "summary": False,
                        "raw_distance": False,
                        "smooth_distance": False,
                        "cost": False,
                        "slope": False
                    }
    print("EXPERIMENTO AR(2) CON MEDIA Y DISPERSIÓN FLUCTUANTES")
    met_gauss_ar2_1, met_emp_ar2_1 = samples_200_arma(seed=1234, N=200, thr_dist=30, min_seg=80, max_seg=120,
                                                        base_mean=0.3, std_range=(0.3, 1.2), s_thresh=22, given=False, geom=True, visualization=visualization, tail=False)
    
    
    print("EXPERIMENTO AR(2) CON REZAGOS FLUCTUANTES")
    met_gauss_ar2_2, met_emp_ar2_2 = samples_200_arma(seed=1234, N=200, thr_dist=30,
                                                        random_phi=True, min_seg=80, max_seg=120,  base_mean=0.5, 
                                                        random_mean=False, base_std=0.5, random_std=True, s_thresh=40, std_range=(0.3, 1.2), given=False, geom=True, visualization=visualization, tail=False)
    
  
    
    np.savez(
                "Resultados/metricas_opt2_suavizado.npz",
                M1=met_gauss_ar2_1,
                M2=met_emp_ar2_1,
                M3=met_gauss_ar2_2,
                M4=met_emp_ar2_2
            )
    
    
    data = np.load("Resultados/metricas_opt2_suavizado.npz")

    generar_informe(data)
    generar_resumen(data)


    
    '''

    start = time.time()
    
    # Listas para acumular métricas de cada dataset
    metricas_kliep = []
    metricas_emp = []
    
    for i in range(5):
        print(f"\n{'='*60}")
        print(f"Procesando dataset {i+1}/5")
        print(f"{'='*60}")
        
        dataset1, cps_ar, outliers_ar = ar2_noise(min_seg=80, max_seg=120, base_mean=0.3, std_range=(0.3, 1.2))
    
        met_dataset1_kliep = best_params_sh(
                dataset1,
                cps_ar,
                modo='Kliep',
                penal=True,
                lambda_p=0,
                given=False,
                geom=True,
                visualization={
                    "summary": False,
                    "raw_distance": False,
                    "smooth_distance": False,
                    "cost": False,
                    "slope": False
                }
            )
        
        met_dataset1_emp = best_params_sh(
                dataset1,
                cps_ar,
                modo='Empírico',
                penal=True,
                lambda_p=0,
                given=False,
                geom=True,
                visualization={
                    "summary": False,
                    "raw_distance": False,
                    "smooth_distance": False,
                    "cost": False,
                    "slope": False
                }
            )
        
        # Guardar métricas en listas (índice [0] contiene el diccionario de métricas)
        metricas_kliep.append(list(met_dataset1_kliep[0].values()))
        metricas_emp.append(list(met_dataset1_emp[0].values()))
    
    end = time.time()
    
    # Crear DataFrames con las métricas
    columnas = ['Mean Location Error', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'Falsos Positivos', 'Falsos Negativos', 'Verdaderos Positivos']
    
    df_kliep = pd.DataFrame(metricas_kliep, columns=columnas)
    df_emp = pd.DataFrame(metricas_emp, columns=columnas)
    
    # Guardar en Excel
    df_kliep.to_excel("Resultados/experimento_kliep.xlsx", index=False)
    df_emp.to_excel("Resultados/experimento_empirico.xlsx", index=False)
    
    print(f"\n{'='*60}")
    print("Resultados guardados:")
    print(f"  - Resultados/experimento_kliep.xlsx")
    print(f"  - Resultados/experimento_empirico.xlsx")
    print(f"Tiempo total de ejecución: {end - start:.2f} segundos")
    print(f"{'='*60}")


    #print("Métricas KLIEP:", met_dataset1_kliep)
    changepoints = klcpd_detect(
                                dataset1,
                                wnd_dim=3,           # Ventana muy pequeña
                                max_iter=20,         # Solo 20 iteraciones
                                batch_size=256,      # Batches grandes
                                eval_freq=10,        # Evalúa cada 10 iters
                                lambda_ae=0.001,
                                lambda_real=0.1,
                                weight_clip=0.1
                            )
    print("Puntos de cambio detectados:", changepoints)
    print()
    print("Puntos de cambio reales:", cps_ar)
    end = time.time()
    print(f"Tiempo de ejecución: {end - start:.2f} segundos")'''
    


    