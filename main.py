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


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

    def graficar(df, filename_suffix):
        plt.figure(figsize=(10, 6))

        if usar_violin:
            sns.violinplot(
                data=df, x="Columna", y="Valor", hue="Matriz",
                split=True, inner="box", palette="pastel"
            )
        else:
            palette = "pastel"

            sns.boxplot(
                data=df, x="Columna", y="Valor", hue="Matriz",
                palette=palette, width=0.5
            )

            sns.stripplot(
                data=df, x="Columna", y="Valor", hue="Matriz",
                dodge=True, jitter=True, alpha=0.5,
                palette=["black", "gray"], size=3
            )

            handles, labels = plt.gca().get_legend_handles_labels()
            plt.legend(handles[:2], labels[:2], title="Matriz")

        plt.title(title) 
        plt.tight_layout()
        plt.savefig(path + filename_suffix, dpi=300, bbox_inches="tight")
        plt.show()

    idx1 = list(range(1, 4))
    cols1 = columnas[1:4]
    df1 = construir_df(idx1, cols1)
    graficar(df1, ' 1')  

    idx2 = [0]
    cols2 = [columnas[0]]
    df2 = construir_df(idx2, cols2)
    graficar(df2, ' 2')  
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




if __name__ == "__main__":
    
    '''
    print("EXPERIMENTO AR(2) CON MEDIA Y DISPERSIÓN FLUCTUANTES")
    met_gauss_ar2_1, met_emp_ar2_1 = samples_200_arma(seed=1234, N=10, thr_dist=30, min_seg=80, max_seg=120,
                                                        base_mean=0.3, std_range=(0.3, 1.2), s_thresh=22, seed2=None, plot_cond=False, tail=False, given=False, plot_slope=True)
    print('Gaussiana')
    print(met_gauss_ar2_1)
    print('Empírica')
    print(met_emp_ar2_1)
    print()



    np.savez(
                "metricas_opt20.npz",
                M1=met_gauss_ar2_1,
                M2=met_emp_ar2_1,
            )
    
    data = np.load("metricas_opt20.npz")
    
    M1_opt = data["M1"]
    M2_opt = data["M2"]
  
    columnas = ['Mean Location Error', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'Falsos Positivos', 'Falsos Negativos', 'Verdaderos Positivos']
    

    boxplot_comp(met_gauss_ar2_1, met_emp_ar2_1, columnas, "Comparación métricas AR variando media y dispersión", 'Gráficas/Umbral Variable/Comparación_AR_Media_20_Pruebas', usar_violin=True)
    
    #plt.plot(prec_list - M1_opt[1])

    '''
    print("EXPERIMENTO AR(2) CON REZAGOS FLUCTUANTES")
    met_gauss_ar2_2, met_emp_ar2_2 = samples_200_arma(seed=1234, N=15, thr_dist=30,
                                                        random_phi=True, min_seg=80, max_seg=120,  base_mean=0.5, 
                                                        random_mean=False, base_std=0.5, random_std=True, s_thresh=40, std_range=(0.3, 1.2), given=False, plot_slope=True)
    
    '''
    print('Gaussiana')
    print(met_gauss_ar2_2)
    print('Empírica')
    print(met_emp_ar2_2)
    print()

    
    np.savez(
                "metricas_opt2.npz",
                M1=met_gauss_ar2_1,
                M2=met_emp_ar2_1,
                M3=met_gauss_ar2_2,
                M4=met_emp_ar2_2
            )
    
    data = np.load("metricas_opt2.npz")

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
    
    df_ar2_1.to_excel("Métricas_Experimento_AR2_1_Variable.xlsx", index=False)
    df_ar2_2.to_excel("Métricas_Experimento_AR2_2_Variable.xlsx", index=False)
       

    columnas = ['Mean Location Error', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'Falsos Positivos', 'Falsos Negativos', 'Verdaderos Positivos']
    data = np.load("metricas_opt2.npz")

    M1_opt = data["M1"]
    M2_opt = data["M2"]
    M3_opt = data["M3"]
    M4_opt = data["M4"]
    #M5_opt = data["M5"]
    #M6_opt = data["M6"]
    #M7_opt = data["M7"]
    #M8_opt = data["M8"]
    df1 = pd.DataFrame(M1_opt, columns=columnas)
    df2 = pd.DataFrame(M2_opt, columns=columnas)
    df3 = pd.DataFrame(M3_opt, columns=columnas)
    df4 = pd.DataFrame(M4_opt, columns=columnas)

    # Exportar a un solo archivo Excel con múltiples hojas
    with pd.ExcelWriter("metricas_opt2_Ajustado.xlsx") as writer:
        df1.to_excel(writer, sheet_name="M1", index=False)
        df2.to_excel(writer, sheet_name="M2", index=False)
        df3.to_excel(writer, sheet_name="M3", index=False)
        df4.to_excel(writer, sheet_name="M4", index=False)
    '''
    
    #boxplot_comp(M1_opt, M2_opt, columnas, "Comparación métricas AR variando media y dispersión", 'Gráficas/Umbral Variable/Comparación_AR_Media')
    #boxplot_comp(M3_opt, M4_opt, columnas, "Comparación métricas AR variando rezagos", 'Gráficas/Umbral Variable/Comparación_AR_Rezagos')


    #boxplot_comp(M5_opt, M6_opt, columnas, "Comparación métricas ARMA")
    #boxplot_comp(M7_opt, M8_opt, columnas, "Comparación métricas Serie Periódica")

    