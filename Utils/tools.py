import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Algoritmo_Gaussiano.cpd import CPD
from Algoritmo_Empiricas.Empirical_CPD import EmpiricalCPD
from Utils.detection import detect
from Utils.metrics_sup import metrics
from scipy.interpolate import UnivariateSpline

def graficar_mapa_calor(start, end, CPD, espacio, title, path, show=False, object_use=True):
    print(f'Tiempo de ejecución {end-start} segundos')
        
    if object_use:
        print(f'Mejor ventana método gaussiano: {CPD.window}')
        print(f'Mejor retardo método gaussiano: {CPD.t}')
    else:
        print(f'Mejor ventana método gaussiano: {CPD[0]}')
        print(f'Mejor retardo método gaussiano: {CPD[1]}')

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



def graficar_dispersion_costo(start, end, CPD, espacio, title, path, show=False, object_use=True):

    print(f'Tiempo de ejecución {end-start} segundos')
    if object_use:
        print(f'Mejor ventana método empírico: {CPD.window}')
    else:
        print(f'Mejor ventana método empírico: {CPD}')


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
    if show:
        plt.show()



def penalized_costs(costs, penals, beta):

    result = {
        key: costs[key] + beta * penals[key]
        for key in costs
    }
    best_key = min(result, key=result.get)
    return best_key, result


def best_params_sh(dataset, cps, path='', name='', s_thresh=0, penal=True, lambda_p=0, plot_cond=True, gauss=True, given=True, plot_slope=False):
    if not gauss:
        CPD_dataset1 = EmpiricalCPD(dataset)
        T = len(dataset)
        start = time.time()
        best_dist, pc_detectados_dataset1, f_costos, penalizaciones = CPD_dataset1. opt_window(
                                                                                                max_w=T//10,
                                                                                                penal=penal,
                                                                                                lambda_p=lambda_p,
                                                                                                join=False)
        end = time.time()
        if plot_cond:
            graficar_dispersion_costo(start, end, CPD_dataset1, f_costos, "Dispersión solo costo "+str(T), path+' empírica solo coste '+name, object_use=False)
            graficar_dispersion_costo(start, end, CPD_dataset1, penalizaciones, "Dispersión penalización "+str(T), path+' empírica solo penalización '+name, object_use=False)

        betha = CPD_dataset1.slope_heuristic_regression(s_thresh, f_costos, penalizaciones, plot=plot_slope, path='Gráficas/Heuristic_Slope/Visualización_Heuristic_Slope '+name, given=given)

        best_key, result = penalized_costs(f_costos, penalizaciones, 2*betha)

        CPD_dataset1.window = best_key
        CPD_dataset1.k_gauss = True
        CPD_dataset1.sigma_filter = round(np.sqrt(CPD_dataset1.window))
        d = CPD_dataset1.distancias()

        pc_detectados_dataset1 = detect(d, CPD_dataset1.window, alpha=0.05, thr=0)
        if plot_cond:
            graficar_dispersion_costo(0, 0, best_key, result, "Dispersión costo "+str(T)+" regularización: "+str(round(betha, 4)), 'Gráficas/Heuristic_Slope/coste '+name, object_use=False)
            print(f'Mejor w: {best_key}')

    else:
        CPD_dataset1 = CPD(dataset)
        T = len(dataset)
        start = time.time()
        best_dist, pc_detectados_dataset1, f_costos, penalizaciones = CPD_dataset1.opt_window_t(max_w=T//10, penal=penal, lambda_p = lambda_p, join=False)
        end = time.time()

        if plot_cond:
            graficar_mapa_calor(start, end, CPD_dataset1, f_costos, "Mapa de calor solo costo "+str(T), path+' gaussiana solo coste '+name)
            graficar_mapa_calor(start, end, CPD_dataset1, penalizaciones, "Mapa de calor penalización "+str(T), path+' gaussiana solo penalización '+name)

        betha = CPD_dataset1.slope_heuristic_regression(s_thresh, f_costos, penalizaciones, plot=plot_slope, path='Gráficas/Heuristic_Slope/Visualización_Heuristic_Slope '+name, given=given)

        best_key, result = penalized_costs(f_costos, penalizaciones, 2*betha)

        CPD_dataset1.window, CPD_dataset1.t = best_key
        CPD_dataset1.k_gauss = True
        CPD_dataset1.sigma_filter = round(np.sqrt(CPD_dataset1.window))
        d = CPD_dataset1.distancias()

        pc_detectados_dataset1 = detect(d, CPD_dataset1.window, alpha=0.05, thr=0)
        if plot_cond:
            graficar_mapa_calor(0, 0, best_key, result, "Mapa de calor costo "+str(T)+" regularización: "+str(round(betha, 4)), 'Gráficas/Heuristic_Slope/coste '+name, object_use=False)
            print(f'Mejor w: {best_key[0]} y mejor t: {best_key[1]}') 

    return metrics(cps, pc_detectados_dataset1, 30, T)



def detectar_codo(x, y):
    # Normalizar (importante para estabilidad numérica)
    x = (x - x.min()) / (x.max() - x.min())
    y = (y - y.min()) / (y.max() - y.min())

    # Recta entre extremos
    p1 = np.array([x[0], y[0]])
    p2 = np.array([x[-1], y[-1]])

    def distancia(p, a, b):
        return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

    distancias = np.array([distancia(np.array([x[i], y[i]]), p1, p2) for i in range(len(x))])
    
    idx_codo = np.argmax(distancias)
    return idx_codo


def hallar_pendiente(penals_arr, costs_arr, s_thresh=None, given=True):
    x = np.asarray(penals_arr)
    y = np.asarray(costs_arr)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if given:
        thresh = s_thresh
    else:
        idx_codo = detectar_codo(x, y)
        thresh = x[idx_codo]
        print(f'CODO: {thresh}')

    # Filtrar
    mask = x >= thresh
    x_filtrado = x[mask]
    y_filtrado = y[mask]

    if len(x_filtrado) < 2:
        raise ValueError("No hay suficientes puntos para regresión.")

    m, _ = np.polyfit(x_filtrado, y_filtrado, 1)
    return abs(m)

'''

def fit_spline(x, y, s=None, plot=False, num_points=500):
    idx = np.argsort(x)
    x_sorted = np.array(x)[idx]
    y_sorted = np.array(y)[idx]

    spline = UnivariateSpline(x_sorted, y_sorted, s=s)

    if plot:
        x_dense = np.linspace(x_sorted.min(), x_sorted.max(), num_points)
        y_dense = spline(x_dense)

        plt.figure()
        plt.scatter(x_sorted, y_sorted, label="Datos", alpha=0.7)
        plt.plot(x_dense, y_dense, label="Spline", linewidth=2)

        plt.title(f"Ajuste spline (s={s})")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.grid(True)
        plt.show()

    return spline, x_sorted

def compute_derivatives(spline, x):
    d1 = spline.derivative(n=1)(x)
 
    return d1

def find_max_derivative_jump(x, d1):
    x = np.asarray(x)
    d1 = np.asarray(d1)

    # diferencias absolutas entre derivadas consecutivas
    diff = np.abs(np.diff(d1[20:-5]))

    if len(diff) == 0:
        return None

    idx = np.argmax(diff)

    return x[idx + 1], d1[idx + 1],
 

def derivada_empirica(x, y, normalize=False):

    x = np.asarray(x[20:])
    y = np.asarray(y[20:])

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    dx = np.diff(x)
    dy = np.diff(y)

    dx = np.where(dx == 0, 1e-12, dx)

    d1 = dy / dx 

    d_diff = np.abs(np.diff(d1))  

    if normalize:
        d_diff = d_diff / (np.abs(d1[:-1]) + 1e-8)

    if len(d_diff) == 0:
        return None

    idx_jump = np.argmax(d_diff)
    start_idx = np.where(x==22)[0][0]

    x_tail = x[start_idx:]
    y_tail = y[start_idx:]

    m, b = np.polyfit(x_tail, y_tail, 1)

    return x[idx_jump + 2], abs(m)
       
def detectar_codo(penals_arr, costs_arr, s=None, spline=False):
    
    if spline:
        spline, x = fit_spline(penals_arr, costs_arr, plot=True)
        d1 = compute_derivatives(spline, x)
        x_codo, deriv_codo = find_max_derivative_jump(x, d1)
    else:
        x_codo, deriv_codo = derivada_empirica(penals_arr, costs_arr)

    print(f'Punto umbral: {x_codo}')
    return deriv_codo
'''