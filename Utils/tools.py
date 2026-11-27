import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from Algoritmo_Gaussiano.cpd import CPD
from Algoritmo_Empiricas.Empirical_CPD import EmpiricalCPD
from Utils.detection import detect
from Utils.metrics_sup import metrics


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


def graficar_mapa_calor(start, end, CPD, espacio, title, path, show=False, object_use=True, best_key=None, save_fig=True):
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
    if save_fig:
        fig.savefig(path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig

def graficar_dispersion_costo(
    start,
    end,
    CPD,
    espacio,
    title,
    path,
    show=False,
    object_use=True,
    best_key=None,
    save_fig=True
):
    print_toolbox_section(
        "Dispersion of Cost"
    )

    print_toolbox_item(
        "Execution Time",
        f"{end - start:.4f} s"
    )

    if object_use:
        print_toolbox_item(
            "Best Window",
            CPD.window
        )
    else:
        print_toolbox_item(
            "Best Window",
            CPD
        )

    ws_emp = np.asarray(
        sorted(espacio.keys())
    )

    costs_emp = np.asarray(
        [
            espacio[w]
            for w in ws_emp
        ],
        dtype=np.float64
    )

    sns.set_theme(
        style="whitegrid",
        context="talk"
    )

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.scatter(
        ws_emp,
        costs_emp,
        s=30,
        alpha=0.85
    )

    # Resaltar la ventana finalmente seleccionada.
    if best_key is not None:
        if isinstance(
            best_key,
            (
                tuple,
                list,
                np.ndarray
            )
        ):
            best_w = best_key[0]
        else:
            best_w = best_key

        selected = np.flatnonzero(
            ws_emp == best_w
        )

        if selected.size > 0:
            index = selected[0]

            ax.scatter(
                ws_emp[index],
                costs_emp[index],
                s=110,
                facecolors="none",
                edgecolors="red",
                linewidths=2.2,
                label=f"Selected Window: {best_w}"
            )

    # Estos son índices del arreglo, no posiciones
    # del eje x.
    max_step = 10

    step_x = min(
        max_step,
        max(
            1,
            len(ws_emp) // 10
        )
    )

    tick_indices = np.unique(
        np.concatenate(
            (
                np.arange(
                    0,
                    len(ws_emp),
                    step_x
                ),
                np.array(
                    [len(ws_emp) - 1]
                )
            )
        )
    )

    # Las posiciones reales de los ticks deben ser
    # los valores de ventana correspondientes.
    tick_values = ws_emp[
        tick_indices
    ]

    ax.set_xticks(
        tick_values
    )

    ax.set_xticklabels(
        [
            str(value)
            for value in tick_values
        ],
        rotation=45,
        ha="right"
    )

    label_size = (
        10
        if len(ws_emp) <= 20
        else 8
        if len(ws_emp) <= 40
        else 6
    )

    ax.tick_params(
        axis="x",
        labelsize=label_size
    )

    ax.set_xlabel(
        "Window Size (w)"
    )

    ax.set_ylabel(
        "Cost"
    )

    ax.set_title(
        title
    )

    ax.grid(
        alpha=0.3
    )

    if best_key is not None:
        ax.legend(
            fontsize=9
        )

    fig.tight_layout()

    if save_fig and path is not None:
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight"
        )

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
    path="Gráficas/",
    name="",
    s_thresh=0,
    penal=True,
    modo="Gaussiano",
    given=True,
    geom=True,
    mean_gauss=True,
    visualization=None,
    thr_dist=30,
    min_w=None,
    max_w=None,
    smooth_distances=True,
    return_cps_list=False,
    t_given=None,
    temporal_comp=True,
    max_comp=False,
    min_comp=False,
    save_fig=True,
    max_bandwidth=False,
    prominence_ratio=0.05
):
    T = len(dataset)

    if visualization is None:
        visualization = {}

    if min_w is None:
        min_w = max(
            9,
            T // 100
        )

    if max_w is None:
        max_w = T // 10

    if modo not in {
        "Gaussiano",
        "Empírico"
    }:
        raise ValueError(
            "modo debe ser 'Gaussiano' o 'Empírico'."
        )


    summary_enabled = visualization.get(
        "summary",
        False
    )
    raw_enabled = visualization.get(
        "raw_distance",
        False
    )
    smooth_enabled = visualization.get(
        "smooth_distance",
        False
    )
    cost_enabled = visualization.get(
        "cost",
        False
    )
    slope_enabled = visualization.get(
        "slope",
        False
    )

    figuras = []
    figuras_generadas = []

    def register_figure(figure, description):
        if figure is not None:
            figuras.append(figure)
            figuras_generadas.append(
                description
            )

    print_toolbox_header(
        f"Optimización {modo}"
    )

    print_toolbox_item(
        "Tamaño de serie",
        T
    )
    print_toolbox_item(
        "Penalización",
        penal
    )

    if penal:
        print_toolbox_item(
            "Codo dado",
            given
        )
        print_toolbox_item(
            "Codo geométrico",
            geom
        )

    if modo == "Gaussiano":
        CPD_dataset1 = CPD(
            dataset,
            smooth=smooth_distances,
            medias=mean_gauss,
            temporal_comp=temporal_comp,
            max_comp=max_comp,
            min_comp=min_comp,
            max_bandwidth=max_bandwidth,
            prominence_ratio=prominence_ratio
        )
    else:
        CPD_dataset1 = EmpiricalCPD(
            dataset,
            smooth=smooth_distances,
            max_bandwidth=max_bandwidth,
            prominence_ratio=prominence_ratio
        )

    start = time.time()

    if modo == "Gaussiano":
        if penal:
            (
                _,
                _,
                f_costos,
                penalizaciones
            ) = CPD_dataset1.opt_window_t(
                max_w=max_w,
                min_w=min_w,
                penal=True,
                join=False,
                t_given=t_given
            )
        else:
            (
                _,
                _,
                f_costos
            ) = CPD_dataset1.opt_window_t(
                max_w=max_w,
                min_w=min_w,
                penal=False,
                join=False,
                t_given=t_given
            )

    else:
        if penal:
            (
                _,
                _,
                f_costos,
                penalizaciones
            ) = CPD_dataset1.opt_window(
                min_w=min_w,
                max_w=max_w,
                penal=True,
                join=False
            )
        else:
            (
                _,
                _,
                f_costos
            ) = CPD_dataset1.opt_window(
                min_w=min_w,
                max_w=max_w,
                penal=False,
                join=False
            )

    end = time.time()

    beta = None
    gamma = None
    result = None
    fig_slope = None

    if penal:
        beta, fig_slope = (
            CPD_dataset1.slope_heuristic_regression(
                s_thresh,
                f_costos,
                penalizaciones,
                plot=slope_enabled,
                path=(
                    path
                    + "Heuristic_Slope/"
                    + "Visualización_Heuristic_Slope "
                    + name
                ),
                given=given,
                geom=geom,
                return_fig=True,
                save_fig=save_fig
            )
        )

        gamma = 2.0 * beta

        best_key, result = penalized_costs(
            f_costos,
            penalizaciones,
            gamma
        )

        register_figure(
            fig_slope,
            "Slope heuristic"
        )

    else:
        best_key = min(
            f_costos,
            key=lambda key: (
                f_costos[key],
                key
            )
        )

    # Se asignan únicamente los parámetros escogidos por
    # el criterio final, no los provisionales retornados
    # por el optimizador.
    if modo == "Gaussiano":
        (
            CPD_dataset1.window,
            CPD_dataset1.t
        ) = (
            int(best_key[0]),
            int(best_key[1])
        )

    else:
        CPD_dataset1.window = int(
            best_key
        )

        CPD_dataset1.last_distance_raw = None
        CPD_dataset1.last_distance_smooth = None

    # Se calcula una sola vez la curva correspondiente
    # a los parámetros definitivamente seleccionados.
    d = CPD_dataset1.distancias()

    pc_detectados_dataset1 = detect(
        d,
        CPD_dataset1.window,
        thr=0,
        prominence_ratio=prominence_ratio
    )

    if cost_enabled:

        use_heatmap = (
            modo == "Gaussiano"
            and t_given is None
        )

        if use_heatmap:
            fig_cost = graficar_mapa_calor(
                start,
                end,
                CPD_dataset1,
                f_costos,
                (
                    "Heat Map of Cost "
                ),
                (
                    path
                    + modo
                    + "/ Gaussian only cost "
                    + name
                ),
                show=False,
                best_key=best_key,
                save_fig=save_fig
            )

            register_figure(
                fig_cost,
                "Heat Map of Cost"
            )

            if penal:
                fig_penalty = graficar_mapa_calor(
                    start,
                    end,
                    CPD_dataset1,
                    penalizaciones,
                    (
                        "Heat Map of Penalty "
                    ),
                    (
                        path
                        + modo
                        + "/ Gaussian only penalty "
                        + name
                    ),
                    show=False,
                    best_key=best_key,
                    save_fig=save_fig
                )
                    
                fig_regularized = graficar_mapa_calor(
                        0,
                        0,
                        CPD_dataset1,
                        result,
                        (
                            "Heat Map of Regularized Cost "
                            + "; gamma="
                            + str(round(gamma, 4))
                        ),
                        (
                            path
                            + "Heuristic_Slope/cost "
                            + name
                        ),
                        show=False,
                        best_key=best_key,
                        save_fig=save_fig
                    )

                register_figure(
                        fig_penalty,
                        "Heat Map of Penalty"
                    )
                register_figure(
                        fig_regularized,
                        "Heat Map of Regularized Cost"
                    )

        else:
            if modo == "Gaussiano":
                plot_best_key = int(
                    best_key[0]
                )

                plot_costs = {
                    key[0]: value
                    for key, value
                    in f_costos.items()
                }

                if penal:
                    plot_penalties = {
                        key[0]: value
                        for key, value
                        in penalizaciones.items()
                    }

                    plot_result = {
                        key[0]: value
                        for key, value
                        in result.items()
                    }

            else:
                plot_best_key = int(
                    best_key
                )
                plot_costs = f_costos

                if penal:
                    plot_penalties = penalizaciones
                    plot_result = result
                
            fig_cost = graficar_dispersion_costo(
                                                        start,
                                                        end,
                                                        CPD_dataset1,
                                                        plot_costs,
                                                        (
                                                            "Dispersion only cost "
                                                        ),
                                                        (
                                                            path
                                                            + modo
                                                            + "/ "
                                                            + modo
                                                            + " only cost "
                                                            + name
                                                            if path is not None
                                                            else None
                                                        ),
                                                        show=False,
                                                        object_use=True,
                                                        best_key=plot_best_key,
                                                        save_fig=save_fig
                                                    )

            register_figure(
                    fig_cost,
                    "Wasserstein Distance Unsmoothed"
            )

            if penal:

                fig_penalty = graficar_dispersion_costo(
                                                            start,
                                                            end,
                                                            CPD_dataset1,
                                                            plot_penalties,
                                                            (
                                                                "Dispersion only penalty "
                                                            ),
                                                            (
                                                                path
                                                                + modo
                                                                + "/ "
                                                                + modo
                                                                + " only penalty "
                                                                + name
                                                                if path is not None
                                                                else None
                                                            ),
                                                            show=False,
                                                            object_use=True,
                                                            best_key=plot_best_key,
                                                            save_fig=save_fig
                                                        )

                fig_regularized = (
                    graficar_dispersion_costo(
                        0,
                        0,
                        plot_best_key,
                        plot_result,
                        (
                            "Dispersion regularized cost "
                            + "; gamma="
                            + str(round(gamma, 4))
                        ),
                        (
                            path
                            + "Heuristic_Slope/cost "
                            + name
                        ),
                        show=False,
                        object_use=False,
                        best_key=plot_best_key,
                        save_fig=save_fig
                    )
                )

                register_figure(
                    fig_penalty,
                    "Dispersion only penalty"
                )
                register_figure(
                    fig_regularized,
                    "Dispersion regularized"
                )

    if raw_enabled:
        fig_raw = CPD_dataset1.plot_distancias(
            suavizada=False,
            title=(
                "Wasserstein Distance "
                "Unsmoothed "
            ),
            path=(
                path
                + modo
                + "/ "
                + modo
                + " Unsmoothed "
                + name
            ),
            show=False,
            save_fig=save_fig
        )

        register_figure(
            fig_raw,
            "Wasserstein Distance Unsmoothed"
        )

    if smooth_enabled:
        fig_smooth = CPD_dataset1.plot_distancias(
            suavizada=True,
            title=(
                "Wasserstein Distance smoothed"
            ),
            path=(
                path
                + modo
                + "/ "
                + modo
                + " smoothed "
                + name
            ),
            show=False,
            save_fig=save_fig
        )

        register_figure(
            fig_smooth,
            "Wasserstein Distance smoothed"
        )

    if summary_enabled:
        fig_resumen = CPD_dataset1.plot_resumen(
            cps,
            pc_detectados_dataset1,
            title=(
                " General Summary of the Change Point Detection Algorithm "
            ),
            path=(
                path
                + modo
                + "/ resumen "
                + modo
                + " "
                + name
            ),
            show=False,
            save_fig=save_fig
        )

        register_figure(
            fig_resumen,
            "General Summary of the Change Point Detection Algorithm"
        )

    print_toolbox_section(
        "Selección final"
    )

    if modo == "Gaussiano":
        print_toolbox_item(
            "Ventana óptima",
            best_key[0]
        )
        print_toolbox_item(
            "Retardo óptimo",
            best_key[1]
        )
    else:
        print_toolbox_item(
            "Ventana óptima",
            best_key
        )

    if penal:
        print_toolbox_item(
            "Penalización mínima",
            beta
        )
        print_toolbox_item(
            "Gamma final",
            gamma
        )

    print_toolbox_item(
        "Tiempo de búsqueda",
        round(end - start, 3)
    )

    print_toolbox_success(
        "Optimización finalizada"
    )

    if figuras_generadas:
        print_toolbox_section(
            "Figuras preparadas"
        )

        for description in figuras_generadas:
            print_toolbox_item(
                "Figura",
                description
            )

    if figuras:
        plt.show()

    metric_result = metrics(
        cps,
        pc_detectados_dataset1,
        thr_dist,
        T
    )

    if return_cps_list:
        return (
            metric_result,
            pc_detectados_dataset1
        )

    return metric_result

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
    return m


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



def generar_tabla_desde_diccionarios(
    lista_dicts,
    lista_nombres,
    columnas_keys,
    nombre_columna_id="Method"
):
    """
    Genera una tabla con una fila por método.

    Cada elemento de lista_dicts debe ser una lista de diccionarios
    con las mismas llaves. Para cada columna se calcula el promedio
    de los valores encontrados en dichos diccionarios.
    """

    filas = []

    for nombre, diccionarios in zip(
        lista_nombres,
        lista_dicts
    ):
        fila = {
            nombre_columna_id: nombre
        }

        for columna in columnas_keys:
            valores = [
                diccionario.get(
                    columna,
                    np.nan
                )
                for diccionario in diccionarios
            ]

            valores = pd.to_numeric(
                pd.Series(valores),
                errors="coerce"
            )

            fila[columna] = (
                valores.mean()
                if valores.notna().any()
                else np.nan
            )

        filas.append(fila)

    return pd.DataFrame(filas)


def grafica_comparativa_wasserstein_serie(comparaciones, title):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 10)
    )

    axes = axes.flatten()

    for ax, (
        serie,
        wasserstein,
        window,
        titulo,
        cps_originales,
        cps_detectados
    ) in zip(axes, comparaciones):

        serie = np.asarray(
            serie,
            dtype=np.float64
        )

        wasserstein = np.asarray(
            wasserstein,
            dtype=np.float64
        )

        cps_originales = np.atleast_1d(
            cps_originales
        )

        cps_detectados = np.atleast_1d(
            cps_detectados
        )

        # Índices de la serie original
        x_serie = np.arange(len(serie))

        # Correspondencia de Wasserstein con la serie original
        x_wasserstein = (
            np.arange(len(wasserstein))
            + window
        )

        # Segundo eje vertical
        ax_wasserstein = ax.twinx()

        ax.plot(
            x_serie,
            serie,
            color="#2878B5",
            linewidth=1.2,
            alpha=0.75,
            label="Serie original"
        )

        ax_wasserstein.plot(
            x_wasserstein,
            wasserstein,
            color="#E87500",
            linewidth=2.2,
            alpha=0.9,
            label="Distancia de Wasserstein"
        )

        # Puntos de cambio originales
        for i, cp in enumerate(cps_originales):
            ax.axvline(
                x=cp,
                color="#C62828",
                linestyle="--",
                linewidth=2.2,
                alpha=0.85,
                label=(
                    "Punto de cambio real"
                    if i == 0
                    else None
                )
            )

        # Puntos de cambio detectados
        for i, cp in enumerate(cps_detectados):
            ax.axvline(
                x=cp,
                color="#2E7D32",
                linestyle=":",
                linewidth=2.4,
                alpha=0.95,
                label=(
                    "Punto de cambio detectado"
                    if i == 0
                    else None
                )
            )

        ax.set_title(
            titulo,
            fontsize=13,
            fontweight="bold"
        )

        ax.set_xlabel("Índice temporal")

        ax.set_ylabel(
            "Serie de tiempo",
            color="#2878B5"
        )

        ax_wasserstein.set_ylabel(
            "Distancia de Wasserstein",
            color="#E87500"
        )

        ax.tick_params(
            axis="y",
            labelcolor="#2878B5"
        )

        ax_wasserstein.tick_params(
            axis="y",
            labelcolor="#E87500"
        )

        ax.set_xlim(
            0,
            len(serie) - 1
        )

        ax.grid(
            alpha=0.25,
            linestyle="--"
        )

        # Combinar las leyendas de ambos ejes
        handles_serie, labels_serie = (
            ax.get_legend_handles_labels()
        )

        handles_wasserstein, labels_wasserstein = (
            ax_wasserstein.get_legend_handles_labels()
        )

        ax.legend(
            handles_serie + handles_wasserstein,
            labels_serie + labels_wasserstein,
            frameon=True,
            fontsize=9,
            loc="best"
        )

    # Ocultar cuadros sobrantes si se proporcionan menos de cuatro
    for ax in axes[len(comparaciones):]:
        ax.set_visible(False)

    fig.suptitle(
        title,
        fontsize=17,
        fontweight="bold",
        y=0.99
    )

    plt.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    plt.show()


def plot_boxplots(
    datos,
    nombres,
    titulo=None,
    ylabel="Value",
    xlabel="Method",
    figsize=(8.4, 4.8),
    palette=None,
    hatches=None,
    show_points=True,
    point_size=16,
    point_alpha=0.50,
    jitter=0.04,
    ylim=None,
    separator_after=2,
    save_path=None,
    dpi=300,
    show=True
):


    if len(datos) != len(nombres):
        raise ValueError(
            "The number of datasets must match the number of labels."
        )

    # ---------------------------------------------------------
    # Limpieza de los datos
    # ---------------------------------------------------------
    datos_limpios = []
    nombres_validos = []
    indices_validos = []

    for indice, (valores, nombre) in enumerate(
        zip(datos, nombres)
    ):
        arr = pd.to_numeric(
            pd.Series(valores),
            errors="coerce"
        ).to_numpy(dtype=float)

        arr = arr[np.isfinite(arr)]

        if arr.size == 0:
            warnings.warn(
                f"Dataset '{nombre}' contains no valid observations."
            )
            continue

        datos_limpios.append(arr)
        nombres_validos.append(nombre)
        indices_validos.append(indice)

    if not datos_limpios:
        raise ValueError(
            "No valid observations are available for plotting."
        )

    # ---------------------------------------------------------
    # Paleta recomendada
    # ---------------------------------------------------------
    default_colors = [
        "#2A9D8F",  # Gaussian optimizado
        "#4C72B0",  # Empirical optimizado
        "#9FD3C7",  # Gaussian con ventana fija
        "#AFC1DE",  # Empirical con ventana fija
        "#D9A441",  # Roerich-KL
        "#9E9E9E"   # Window-RBF
    ]

    if palette is None:
        colores = [
            default_colors[i % len(default_colors)]
            for i in indices_validos
        ]

    elif isinstance(palette, dict):
        colores = [
            palette.get(
                nombre,
                default_colors[j % len(default_colors)]
            )
            for j, nombre in enumerate(nombres_validos)
        ]

    else:
        if len(palette) < len(datos):
            raise ValueError(
                "The palette must contain one color "
                "for each dataset."
            )

        colores = [
            palette[i]
            for i in indices_validos
        ]

    # Patrones opcionales para impresión en escala de grises.
    if hatches is None:
        patrones = [""] * len(datos_limpios)
    else:
        if len(hatches) < len(datos):
            raise ValueError(
                "The hatches list must contain one value "
                "for each dataset."
            )

        patrones = [
            hatches[i]
            for i in indices_validos
        ]

    # ---------------------------------------------------------
    # Estilo general
    # ---------------------------------------------------------
    style = {
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.labelsize": 10.5,
        "axes.titlesize": 11.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5
    }

    with plt.rc_context(style):

        fig, ax = plt.subplots(
            figsize=figsize,
            constrained_layout=True
        )

        positions = np.arange(
            1,
            len(datos_limpios) + 1
        )

        # -----------------------------------------------------
        # Diagramas de caja
        # -----------------------------------------------------
        bp = ax.boxplot(
            datos_limpios,
            positions=positions,
            patch_artist=True,
            widths=0.52,
            whis=1.5,
            showfliers=not show_points,
            medianprops={
                "color": "#202020",
                "linewidth": 1.7
            },
            whiskerprops={
                "color": "#404040",
                "linewidth": 1.1
            },
            capprops={
                "color": "#404040",
                "linewidth": 1.1
            },
            boxprops={
                "edgecolor": "#404040",
                "linewidth": 1.1
            },
            flierprops={
                "marker": "o",
                "markersize": 3.5,
                "markerfacecolor": "#A0A0A0",
                "markeredgecolor": "#404040",
                "markeredgewidth": 0.4,
                "alpha": 0.55
            }
        )

        rng = np.random.default_rng(42)

        for i, (
            valores,
            color,
            patron
        ) in enumerate(
            zip(
                datos_limpios,
                colores,
                patrones
            ),
            start=1
        ):
            box = bp["boxes"][i - 1]

            box.set_facecolor(color)
            box.set_alpha(0.65)
            box.set_hatch(patron)
            box.set_zorder(2)

            if show_points:
                x = i + rng.uniform(
                    -jitter,
                    jitter,
                    size=valores.size
                )

                ax.scatter(
                    x,
                    valores,
                    s=point_size,
                    color=color,
                    alpha=point_alpha,
                    edgecolor="#303030",
                    linewidth=0.35,
                    zorder=3
                )

        # Las medianas quedan por encima de los puntos.
        for median in bp["medians"]:
            median.set_zorder(4)

        # -----------------------------------------------------
        # Ejes y etiquetas
        # -----------------------------------------------------
        ax.set_xticks(positions)
        ax.set_xticklabels(nombres_validos)

        if xlabel is not None:
            ax.set_xlabel(
                xlabel,
                labelpad=7
            )

        ax.set_ylabel(
            ylabel,
            labelpad=7
        )

        if titulo:
            ax.set_title(
                titulo,
                pad=10,
                fontweight="semibold"
            )

        if ylim is not None:
            ax.set_ylim(ylim)

        ax.set_xlim(
            0.5,
            len(datos_limpios) + 0.5
        )

        # Separación entre métodos optimizados y ventana fija.
        if (
            separator_after is not None
            and 0 < separator_after < len(datos_limpios)
        ):
            ax.axvline(
                x=separator_after + 0.5,
                color="#BDBDBD",
                linestyle=":",
                linewidth=0.9,
                alpha=0.85,
                zorder=0
            )

        # -----------------------------------------------------
        # Cuadrícula y bordes
        # -----------------------------------------------------
        ax.set_axisbelow(True)

        ax.grid(
            axis="y",
            color="#D8D8D8",
            linestyle="--",
            linewidth=0.7,
            alpha=0.65
        )

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.spines["left"].set_color("#555555")
        ax.spines["bottom"].set_color("#555555")

        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

        ax.tick_params(
            axis="both",
            width=0.8,
            color="#555555"
        )

        # -----------------------------------------------------
        # Guardado y visualización
        # -----------------------------------------------------
        if save_path is not None:
            fig.savefig(
                save_path,
                dpi=dpi,
                bbox_inches="tight",
                facecolor="white"
            )

        if show:
            plt.show()

    return fig, ax