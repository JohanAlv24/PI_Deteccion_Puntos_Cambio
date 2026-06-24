import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Algoritmo_Gaussiano.cpd import CPD
from Algoritmo_Empiricas.Empirical_CPD import EmpiricalCPD
from Algoritmo_KLIEP.kliep_cpd import KLIEP_CPD
from Utils.detection import detect
from Utils.metrics_sup import metrics
from scipy.interpolate import UnivariateSpline


def _formatear_valor(value, precision=4):
    if isinstance(value, bool):
        return "Sí" if value else "No"
    if isinstance(value, (float, np.floating)):
        return f"{value:.{precision}f}"
    return value


def print_toolbox_header(title, width=72):
    line = "═" * width
    print(f"\n{line}")
    print(f"{title.center(width)}")
    print(f"{line}")


def print_toolbox_section(title, width=72):
    print(f"\n{title}")
    print("─" * len(title))


def print_toolbox_item(label, value, indent=4, precision=4):
    value = _formatear_valor(value, precision=precision)
    print(f"{' ' * indent}{label:<28}: {value}")


def print_toolbox_success(message, indent=4):
    print(f"{' ' * indent}[OK] {message}")


def print_toolbox_warning(message, indent=4):
    print(f"{' ' * indent}[WARN] {message}")


def print_metrics_block(title, metric_dict, indent=4, precision=4):
    print_toolbox_section(title)
    for key, value in metric_dict.items():
        print_toolbox_item(str(key), value, indent=indent, precision=precision)


def graficar_mapa_calor(start, end, CPD, espacio, title, path, show=False, object_use=True, best_key=None):
    print_toolbox_section("Mapa de calor")
    print_toolbox_item("Tiempo de ejecución", f"{end-start:.4f} s")

    if object_use:
        print_toolbox_item("Mejor ventana", CPD.window)
        print_toolbox_item("Mejor retardo", CPD.t)
    else:
        print_toolbox_item("Mejor ventana", CPD[0])
        print_toolbox_item("Mejor retardo", CPD[1])

    ws = sorted(set(k[0] for k in espacio.keys()))
    ts = sorted(set(k[1] for k in espacio.keys()))
    cost_matrix = np.full((len(ws), len(ts)), np.nan)

    ws_index = {w: i for i, w in enumerate(ws)}
    ts_index = {t: j for j, t in enumerate(ts)}

    for (w, t), cost in espacio.items():
        i = ws_index[w]
        j = ts_index[t]
        cost_matrix[i, j] = cost

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(cost_matrix, aspect='auto', origin='lower', cmap='viridis')
    fig.colorbar(im, ax=ax, label="Costo")

    max_step = 10
    step_x = min(max_step, max(1, len(ts) // 10))
    step_y = min(max_step, max(1, len(ws) // 10))

    x_tick_positions = np.unique(np.concatenate((np.arange(0, len(ts), step_x), [len(ts) - 1])))
    y_tick_positions = np.unique(np.concatenate((np.arange(0, len(ws), step_y), [len(ws) - 1])))

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([ts[i] for i in x_tick_positions], rotation=45, ha='right')
    ax.set_yticks(y_tick_positions)
    ax.set_yticklabels([ws[i] for i in y_tick_positions])

    label_size = 10 if max(len(ts), len(ws)) <= 20 else 8 if max(len(ts), len(ws)) <= 40 else 6
    ax.tick_params(axis='x', labelsize=label_size)
    ax.tick_params(axis='y', labelsize=label_size)

    if best_key is None and object_use and hasattr(CPD, "window") and hasattr(CPD, "t"):
        best_key = (CPD.window, CPD.t)

    if best_key is not None and isinstance(best_key, (tuple, list, np.ndarray)) and len(best_key) >= 2:
        w_best, t_best = best_key[0], best_key[1]
        if w_best in ws_index and t_best in ts_index:
            ax.scatter(
                ts_index[t_best],
                ws_index[w_best],
                s=110,
                marker='o',
                facecolors='none',
                edgecolors='red',
                linewidths=2.2
            )

    ax.set_xlabel("t")
    ax.set_ylabel("w")
    ax.set_title(title)

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def graficar_dispersion_costo(start, end, CPD, espacio, title, path, show=False, object_use=True, best_key=None):
    print_toolbox_section("Dispersión de costo")
    print_toolbox_item("Tiempo de ejecución", f"{end-start:.4f} s")

    if object_use:
        print_toolbox_item("Mejor ventana", CPD.window)
    else:
        print_toolbox_item("Mejor ventana", CPD)

    ws_emp = np.array(sorted(espacio.keys()))
    costs_emp = np.array([espacio[w] for w in ws_emp])

    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.scatter(ws_emp, costs_emp, s=30, alpha=0.85)

    if best_key is not None:
        if isinstance(best_key, (tuple, list, np.ndarray)):
            best_w = best_key[0]
        else:
            best_w = best_key
        if np.any(ws_emp == best_w):
            idx = np.where(ws_emp == best_w)[0][0]
            ax.scatter(ws_emp[idx], costs_emp[idx], s=110, facecolors='none', edgecolors='red', linewidths=2.2)

    max_step = 10
    step_x = min(max_step, max(1, len(ws_emp) // 10))
    x_tick_positions = np.unique(np.concatenate((np.arange(0, len(ws_emp), step_x), [len(ws_emp) - 1])))

    ax.set_xticks(x_tick_positions)
    ax.set_xticklabels([ws_emp[i] for i in x_tick_positions], rotation=45, ha='right')
    label_size = 10 if len(ws_emp) <= 20 else 8 if len(ws_emp) <= 40 else 6
    ax.tick_params(axis='x', labelsize=label_size)

    ax.set_xlabel("Tamaño de ventana (w)")
    ax.set_ylabel("Costo")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig


def penalized_costs(costs, penals, beta):
    result = {
        key: costs[key] + beta * penals[key]
        for key in costs
    }
    best_key = min(result, key=result.get)
    return best_key, result


def best_params_sh(
    dataset,
    cps,
    path='Gráficas/',
    name='',
    s_thresh=0,
    penal=True,
    lambda_p=0,
    modo="Gaussiano",
    given=True,
    geom=True,
    visualization=None,
    thr_dist=30,
    min_w = None,
    max_w = None
):
    T = len(dataset)

    if visualization is None:
        visualization = {}

    if min_w is None:
        min_w = T // 100

    if max_w is None:
        max_w = T // 10

    summary_enabled = visualization.get("summary", False)
    raw_enabled = visualization.get("raw_distance", False)
    smooth_enabled = visualization.get("smooth_distance", False)
    cost_enabled = visualization.get("cost", False)
    slope_enabled = visualization.get("slope", False)

    figuras = []
    figuras_generadas = []


    print_toolbox_header(f"Optimización {modo}")

    if modo == "Gaussiano":

        CPD_dataset1 = CPD(dataset)

        T = len(dataset)

        print_toolbox_item("Tamaño de serie", T)
        print_toolbox_item("Penalización", penal)
        print_toolbox_item("Codo dado", given)
        print_toolbox_item("Codo geométrico", geom)

        start = time.time()

        best_dist, pc_detectados_dataset1, f_costos, penalizaciones = CPD_dataset1.opt_window_t(
            max_w = max_w,
            min_w = min_w,
            penal=penal,
            lambda_p=lambda_p,
            join=False
        )

        end = time.time()

        betha, fig_slope = CPD_dataset1.slope_heuristic_regression(
            s_thresh,
            f_costos,
            penalizaciones,
            plot=slope_enabled,
            path='Gráficas/Heuristic_Slope/Visualización_Heuristic_Slope ' + name,
            given=given,
            geom=geom,
            return_fig=True
        )

        if fig_slope is not None:

            figuras.append(fig_slope)

            figuras_generadas.append(
                "Slope heuristic"
            )

        best_key, result = penalized_costs(
            f_costos,
            penalizaciones,
            2 * betha
        )

        CPD_dataset1.window, CPD_dataset1.t = best_key

        CPD_dataset1.smooth = True

        # smoothing sigma parameter removed

        d = CPD_dataset1.distancias()

        pc_detectados_dataset1 = detect(
            d,
            CPD_dataset1.window,
            alpha=0.05,
            thr=0
        )

        if cost_enabled:

            fig1 = graficar_mapa_calor(
                start,
                end,
                CPD_dataset1,
                f_costos,
                "Mapa de calor solo costo " + str(T),
                path + modo + '/ gaussiana solo coste ' + name,
                show=False,
                best_key=best_key
            )

            fig2 = graficar_mapa_calor(
                start,
                end,
                CPD_dataset1,
                penalizaciones,
                "Mapa de calor penalización " + str(T),
                path + modo + '/ gaussiana solo penalización ' + name,
                show=False,
                best_key=best_key
            )

            fig3 = graficar_mapa_calor(
                0,
                0,
                CPD_dataset1,
                result,
                "Mapa de calor costo " + str(T) + " regularización: " + str(round(betha, 4)),
                'Gráficas/Heuristic_Slope/coste ' + name,
                show=False,
                best_key=best_key
            )

            figuras.extend([fig1, fig2, fig3])

            figuras_generadas.extend([
                "Mapa de calor costo",
                "Mapa de calor penalización",
                "Mapa de calor regularizado"
            ])

        if raw_enabled:

            fig_raw = CPD_dataset1.plot_distancias(
                suavizada=False,
                title="Distancia de Wasserstein sin suavizar " + str(T),
                path=path + modo + '/ gaussiana sin suavizar ' + name,
                show=False
            )

            figuras.append(fig_raw)

            figuras_generadas.append(
                "Distancia de Wasserstein sin suavizar"
            )

        if smooth_enabled:

            fig_smooth = CPD_dataset1.plot_distancias(
                suavizada=True,
                title="Distancia de Wasserstein suavizada " + str(T),
                path=path + modo + '/ gaussiana suavizada ' + name,
                show=False
            )

            figuras.append(fig_smooth)

            figuras_generadas.append(
                "Distancia de Wasserstein suavizada"
            )

        if summary_enabled:

            fig_resumen = CPD_dataset1.plot_resumen(
                cps,
                pc_detectados_dataset1,
                title="Resumen general " + str(T),
                path=path + modo + '/ resumen gaussiano ' + name,
                show=False
            )

            figuras.append(fig_resumen)

            figuras_generadas.append(
                "Resumen general"
            )

        print_toolbox_section("Selección final")

        print_toolbox_item(
            "Ventana óptima",
            best_key[0]
        )

        print_toolbox_item(
            "Retardo óptimo",
            best_key[1]
        )

        print_toolbox_item(
            "Beta",
            betha
        )

        print_toolbox_success(
            "Optimización finalizada"
        )

    else:
        T = len(dataset)

        print_toolbox_item("Tamaño de serie", T)
        print_toolbox_item("Penalización", penal)
        print_toolbox_item("Codo dado", given)
        print_toolbox_item("Codo geométrico", geom)


        if modo == "Empírico":
            CPD_dataset1 = EmpiricalCPD(dataset)
        
            start = time.time()

            best_dist, pc_detectados_dataset1, f_costos, penalizaciones = CPD_dataset1.opt_window(
                min_w = min_w,
                max_w = max_w,
                penal=penal,
                lambda_p=lambda_p,
                join=False
            )

            end = time.time()

        if modo == 'Kliep':
            
            CPD_dataset1 = KLIEP_CPD(dataset)

            start = time.time()

            best_dist, pc_detectados_dataset1, f_costos, penalizaciones = CPD_dataset1.opt_window(
                min_w=min_w,
                max_w=max_w,
                penal=penal,
                lambda_p=lambda_p,
                join=False
             )

            end = time.time()

        betha, fig_slope = CPD_dataset1.slope_heuristic_regression(
            s_thresh,
            f_costos,
            penalizaciones,
            plot=slope_enabled,
            path='Gráficas/Heuristic_Slope/Visualización_Heuristic_Slope ' + name,
            given=given,
            geom=geom,
            return_fig=True
        )

        if fig_slope is not None:
            figuras.append(fig_slope)
            figuras_generadas.append("Slope heuristic")

        best_key, result = penalized_costs(
            f_costos,
            penalizaciones,
            2 * betha
        )

        CPD_dataset1.window = best_key
        if modo == 'Empírico':
            CPD_dataset1.smooth = True
            # smoothing sigma parameter removed

            d = CPD_dataset1.distancias()

            pc_detectados_dataset1 = detect(
                d,
                CPD_dataset1.window,
                alpha=0.05,
                thr=0,
                emp=True
            )
        
        if modo == 'Kliep':
            pc_detectados_dataset1 = CPD_dataset1.distancias(return_cps=True)

        if cost_enabled:

            fig1 = graficar_dispersion_costo(
                start,
                end,
                CPD_dataset1,
                f_costos,
                "Dispersión solo costo " + str(T),
                path + modo + '/ ' + modo + ' solo coste ' + name,
                show=False,
                object_use=True
            )

            fig2 = graficar_dispersion_costo(
                start,
                end,
                CPD_dataset1,
                penalizaciones,
                "Dispersión penalización " + str(T),
                path + modo + '/ ' + modo + ' solo penalización ' + name,
                show=False,
                object_use=True
            )

            fig3 = graficar_dispersion_costo(
                0,
                0,
                best_key,
                result,
                "Dispersión costo " + str(T) + " regularización: " + str(round(betha, 4)),
                'Gráficas/Heuristic_Slope/coste ' + name,
                show=False,
                object_use=False,
                best_key=best_key
            )

            figuras.extend([fig1, fig2, fig3])

            figuras_generadas.extend([
                "Dispersión costo",
                "Dispersión penalización",
                "Dispersión regularizada"
            ])

        if raw_enabled:

            fig_raw = CPD_dataset1.plot_distancias(
                suavizada=False,
                title="Distancia de Wasserstein sin suavizar " + str(T),
                path=path + modo + '/ ' + modo + ' sin suavizar ' + name,
                show=False
            )

            figuras.append(fig_raw)

            figuras_generadas.append(
                "Distancia de Wasserstein sin suavizar"
            )

        if smooth_enabled:

            fig_smooth = CPD_dataset1.plot_distancias(
                suavizada=True,
                title="Distancia de Wasserstein suavizada " + str(T),
                path=path + modo + '/ ' + modo + ' suavizada ' + name,
                show=False
            )

            figuras.append(fig_smooth)

            figuras_generadas.append(
                "Distancia de Wasserstein suavizada"
            )

        if summary_enabled:

            fig_resumen = CPD_dataset1.plot_resumen(
                cps,
                pc_detectados_dataset1,
                title="Resumen general " + str(T),
                path=path + modo + '/ resumen ' + modo + ' ' + name,
                show=False
            )

            figuras.append(fig_resumen)

            figuras_generadas.append(
                "Resumen general"
            )

        print_toolbox_section("Selección final")

        print_toolbox_item(
            "Ventana óptima",
            best_key
        )

        print_toolbox_item(
            "Beta",
            betha
        )

        print_toolbox_success(
            "Optimización finalizada"
        )

    

    if figuras_generadas:

        print_toolbox_section("Figuras preparadas")

        for elemento in figuras_generadas:

            print_toolbox_item(
                "Figura",
                elemento
            )

    if figuras:
        plt.show()

    return metrics(
        cps,
        pc_detectados_dataset1,
        thr_dist,
        T
    )

def hallar_pendiente(penals_arr, costs_arr, s_thresh=None, given=True, plot=False):

    x = np.asarray(penals_arr)
    y = np.asarray(costs_arr)
    
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if given:
        thresh = s_thresh
    else:
        y_smooth = suavizar_media_movil(y)
        idx_codo = detectar_codo(x, y_smooth)
        thresh = x[idx_codo]


    mask = x >= thresh
    x_filtrado = x[mask]
    y_filtrado = y[mask]

    if len(x_filtrado) < 2:
        raise ValueError("No hay suficientes puntos para regresión.")

    m, _ = np.polyfit(x_filtrado, y_filtrado, 1)
    return abs(m)


def suavizar_media_movil(y, window=5, preserve_edges=3):
    y = np.asarray(y, dtype=np.float64)

    if len(y) == 0:
        return y

    window = max(1, min(window, len(y)))
    preserve_edges = max(0, min(preserve_edges, len(y)))
    kernel = np.ones(window) / window

    y_smooth = np.convolve(y, kernel, mode='same')

    y_smooth[:preserve_edges] = y[:preserve_edges]
    y_smooth[-preserve_edges:] = y[-preserve_edges:]

    return y_smooth



def detectar_codo(penals_arr, costs_arr, s=None, geom=True):

    x = np.asarray(penals_arr, dtype=float)
    y = np.asarray(costs_arr, dtype=float)

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if len(x) == 0 or len(y) == 0:
        return 0

    x_range = x.max() - x.min()
    y_range = y.max() - y.min()

    if x_range == 0 or y_range == 0:
        return 0

    x_n = (x - x.min()) / x_range
    y_n = (y - y.min()) / y_range


    if geom:

        p1 = np.array([x_n[0], y_n[0]])
        p2 = np.array([x_n[-1], y_n[-1]])

        def distancia(p, a, b):
            return np.abs(np.cross(b - a, a - p)) / np.linalg.norm(b - a)

        distancias = np.array([
            distancia(np.array([x_n[i], y_n[i]]), p1, p2)
            for i in range(len(x_n))
        ])

        idx_codo = int(np.argmax(distancias))


    else:


        a_k = (x_n - 1)**2 + (y_n)**2

        b_k = (x_n)**2 + (y_n - 1)**2

        c_k = (y_n)**2

        denom = a_k + c_k

        # Evitar división por cero
        denom[denom == 0] = np.finfo(float).eps

        f_k = b_k / denom

        idx_codo = int(np.argmax(f_k))

    print_toolbox_section("Detección de codo")

    if geom:
        print_toolbox_item("Método", "Geométrico")
    else:
        print_toolbox_item("Método", "Kneedle")

    print_toolbox_item("Punto umbral", x[idx_codo])

    return idx_codo


'''
def detectar_codo_por_estabilizacion(
    x,
    y,
    q=0.10,
    min_points=5,
    usar_valor_absoluto=True,
    plot=False
):
    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    if len(x) <= min_points:
        raise ValueError("No hay suficientes puntos para detectar el codo.")

    pendientes = []
    x_reg = []

    for i in range(len(x) - min_points):
        x_sub = x[i:]
        y_sub = y[i:]

        m, _ = np.polyfit(x_sub, y_sub, 1)

        if usar_valor_absoluto:
            m = abs(m)

        pendientes.append(m)
        x_reg.append(x[i])

    pendientes = np.array(pendientes)
    x_reg = np.array(x_reg)

    m_min = pendientes.min()
    m_max = pendientes.max()

    banda_sup = m_min + q * (m_max - m_min)

    idx_codo = np.argmax(pendientes <= banda_sup)

    x_codo = x_reg[idx_codo]
    pendiente_codo = pendientes[idx_codo]

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            x_reg,
            pendientes,
            marker='o',
            label='Pendientes iterativas'
        )

        ax.axhline(
            banda_sup,
            linestyle='--',
            label=f'Banda q={q}'
        )

        ax.axvline(
            x_codo,
            linestyle=':',
            linewidth=2,
            label=f'Codo detectado = {x_codo:.2f}'
        )

        ax.scatter(
            x_codo,
            pendiente_codo,
            s=100,
            zorder=5
        )

        ax.set_xlabel("Penalización inicial de regresión")
        ax.set_ylabel("Pendiente absoluta")

        ax.set_title("Estabilización de pendientes")

        ax.grid(True)
        ax.legend()

        plt.show()

    return idx_codo'''