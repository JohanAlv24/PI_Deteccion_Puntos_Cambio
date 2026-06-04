import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from csaps import csaps
import multiprocessing as mp
import scipy.io as scio
from Algoritmo_Empiricas.workers_empirical import init_worker_empirical, evaluate_window_worker
import os

class EmpiricalCPD():
    '''
    X: serie de tiempo
    window: Tamaño de la ventana
    smooth: Condicional para aplicar filtro gaussiano a la curva de distancias de Wasserstein
    '''
    def __init__(self, X, window=0, smooth=True):

        self.Serie = np.asarray(X, dtype=np.float64)

        self.window = int(window)

        self.smooth = bool(smooth)

        self._sorted_unique = None
        self._weights = None

        self.last_distance_raw = None
        self.last_distance_smooth = None


        current_dir = os.path.dirname(os.path.abspath(__file__))
        filter_path = os.path.join(current_dir, "TwoSampConvFilter.mat")

        self.filter = scio.loadmat(filter_path)["filter2"].flatten()

        self.summary_figures = {
            "raw_distance": None,
            "smooth_distance": None,
            "cost_scatter": None,
            "summary": None,
            "slope": None
        }
        

    @staticmethod
    #beta corresponde al tamaño de ventana. Aquí se calcula la distancia de Wasserstein empírica y se retorna la curva rugosa de distancias
    def empirical_cpd(serie, beta):

        X = np.array([
            serie[i:i+beta]
            for i in range(len(serie) - beta + 1)
        ])

        d = []

        for i in range(len(X) - beta):

            p1 = np.sort(X[i])
            p2 = np.sort(X[i+beta])

            d.append(np.mean(np.abs(p1 - p2)))

        return np.array(d)

    #Calcula la distancia de Wasserstein entre ventanas consecutivas no traslapadas de la serie de tiempo y retorna la curva de distancia suavizada
    def distancias(self):

        w = self.window

        d = self.empirical_cpd(self.Serie, w)

        self.last_distance_raw = np.array(d)

        if self.smooth:

            x = np.arange(len(d))

            #d_smooth, smooth_used = csaps(x, d, x)

            self.filter = self.filter[0::int(np.ceil(len(self.filter)/(2*self.window)))]-0.166
            self.filter = self.filter / np.sum(self.filter)
            d_smooth = np.convolve(d, self.filter, mode='same')
            self.last_distance_smooth = np.array(d_smooth)


            return d_smooth

        self.last_distance_smooth = np.array(d)

        return d

    def mle(self):

        sorted_unique = np.sort(np.unique(self.Serie))

        T = len(sorted_unique)

        u = np.arange(1, T + 1)

        weights = 1.0 / ((u - 0.5) * (T - u + 0.5))

        self._sorted_unique = sorted_unique
        self._weights = weights

        return sorted_unique, weights

    def segment_cost_mle(self, start, end, sorted_unique, weights):
        '''
        Calcula el costo de un segmento de la serie temporal bajo el enfoque de máxima verosimilitud
        basado en la entropía de una distribución empírica.
        Los parámetros:
        start, end: índices que delimitan el segmento 
        sorted_unique: valores únicos ordenados de la serie 
        weights: pesos asociados a cada valor (el artículo ya da una formulación para estos pesos)
        '''
        segment = self.Serie[start:end]

        n = len(segment)

        segment_sorted = np.sort(segment)

        counts = np.searchsorted(segment_sorted, sorted_unique, side="right")

        F_hat = counts / n

        eps = 1e-12

        F_hat = np.clip(F_hat, eps, 1 - eps)

        entropy_term = (
            F_hat * np.log(F_hat)
            + (1 - F_hat) * np.log(1 - F_hat)
        )

        return -n * np.sum(entropy_term * weights)

    '''
    La función total_cost recibe la lista de puntos de cambio (change_poins) y aplica la función de coste a cada segmento
    para hallar la función de coste total. En esta función también se añade retorna el factor de penalización pero es integrado 
    en el proceso de optimización
    '''
    def total_cost(self, change_points, penal=True):

        sorted_unique, weights = self.mle()

        total = 0.0

        for i in range(len(change_points) - 1):

            total += self.segment_cost_mle(
                change_points[i],
                change_points[i + 1],
                sorted_unique,
                weights
            )

        if penal:

            penalty = len(change_points)

            return total, penalty

        return total

    '''
    Realiza la búsque exhaustiva del tamaño de ventana. 
    min_w y max_w definen el intervalo de búsqueda para la ventana, penal es un condicional
    para incorporar la penalización y lambda_p es el regularizador que por defecto es -1
    para llevarlo al mismo orden del coste (1/(log(T)*T))
    '''
    def opt_window(
        self,
        min_w=None,
        max_w=None,
        penal=False,
        lambda_p=0,
        join=True
    ):

        T = len(self.Serie)

        if not min_w:
            min_w = 9

        if not max_w:
            max_w = T // 2

        print("\n" + "=" * 70)
        print(" EmpiricalCPD :: Exhaustive Window Optimization")
        print("=" * 70)
        print(f" Serie length                : {T}")
        print(f" Window search interval      : [{min_w}, {max_w}]")
        print(f" Penalized optimization      : {penal}")
        print(f" Gaussian smoothing enabled  : {self.smooth}")
        print("=" * 70)

        windows = np.arange(min_w, max_w + 1, dtype=int)

        tasks = []

        config = {
            "smooth": self.smooth
        }

        for w in windows:
            tasks.append((w, penal, lambda_p, config))

        shared_array = mp.Array('d', self.Serie, lock=False)

        length = len(self.Serie)

        best_cost = np.inf
        best_window = None
        best_cp = None
        best_dist = None

        if join:
            espacio = {}

        f_costos = {}
        penalizaciones = {}

        n_jobs = max(1, mp.cpu_count() - 1)

        print(f" Parallel workers            : {n_jobs}")
        print(f" Total candidate windows     : {len(tasks)}")
        print("-" * 70)

        with mp.Pool(
            processes=n_jobs,
            initializer=init_worker_empirical,
            initargs=(shared_array, length)
        ) as pool:
            
            for idx, result in enumerate(
                pool.imap_unordered(
                    evaluate_window_worker,
                    tasks,
                    chunksize=64
                ),
                start=1
            ):

                if penal:

                    total, penalty, w, cps, distancias = result

                    f_costos[w] = total
                    penalizaciones[w] = penalty

                    cost = total + lambda_p * penalty

                else:

                    cost, w, cps, distancias = result

                if join:
                    espacio[w] = cost

                if cost < best_cost:

                    best_cost = cost
                    best_window = w
                    best_cp = cps
                    best_dist = distancias

                    print(
                        f"[BEST UPDATE] "
                        f"w={w:<4} | "
                        f"cost={best_cost:.6f} | "
                        f"cp={len(best_cp)}"
                    )

                if idx % max(1, len(tasks)//10) == 0:

                    progress = 100 * idx / len(tasks)

                    print(
                        f"[PROGRESS] "
                        f"{idx}/{len(tasks)} "
                        f"({progress:.1f}%)"
                    )

        if best_window is not None:

            self.window = best_window

        print("-" * 70)
        print(" Optimización Finalizada")
        print(f" ventana óptima              : {self.window}")
        # smoothing sigma parameter removed
        print(f" Mejor valor función objetivo        : {best_cost:.6f}")
        print("=" * 70 + "\n")

        if penal:

            if join:
                return best_dist, best_cp, espacio, f_costos, penalizaciones

            return best_dist, best_cp, f_costos, penalizaciones

        return best_dist, best_cp, espacio

    def slope_heuristic_fig(
        self,
        costs,
        penals,
        plot=True,
        path='slope_heuristic',
        s_thresh=0,
        given=True,
        geom=True,
        return_fig=False
    ):

        penal_to_costs = {}

        for key in costs:

            penal = penals[key]
            cost = costs[key]

            if penal not in penal_to_costs:
                penal_to_costs[penal] = []

            penal_to_costs[penal].append(cost)

        penal_min_cost = {
            penal: min(cost_list)
            for penal, cost_list in penal_to_costs.items()
        }

        result = sorted(
            penal_min_cost.items(),
            key=lambda x: x[0]
        )

        penals_sorted = [x[0] for x in result]
        min_costs = [x[1] for x in result]

        fig = None

        if plot:

            from Utils.tools import suavizar_media_movil, detectar_codo

            sns.set_theme(style="whitegrid", context="talk")
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(
                penals_sorted,
                min_costs,
                marker='o',
                linewidth=2.0,
                label='Costo mínimo'
            )

            thresh = None
            idx_codo = None

            if given:
                thresh = s_thresh
            else:
                y_smooth = suavizar_media_movil(
                            np.asarray(min_costs, dtype=np.float64)
                            )
                ax.plot(
                    penals_sorted,
                    y_smooth,
                    linewidth=2.2,
                    linestyle='--',
                    label='Suavizada'
                )
                idx_codo = detectar_codo(np.asarray(penals_sorted, dtype=np.float64), y_smooth, geom=geom)
                thresh = penals_sorted[idx_codo]
  

            if thresh is not None:
                ax.axvline(
                    thresh,
                    linestyle='--',
                    linewidth=2.0,
                    color='tab:red',
                    label=f'Codo detectado = {thresh:.2f}'
                )
                idx_plot = int(
                    np.argmin(
                        np.abs(
                            np.asarray(penals_sorted, dtype=np.float64) - thresh
                        )
                    )
                )
                ax.scatter(
                    penals_sorted[idx_plot],
                    min_costs[idx_plot],
                    s=40,
                    zorder=5,
                    color='tab:red'
                )

            ax.set_xlabel("Penalización")
            ax.set_ylabel("Costo mínimo")
            ax.set_title("Heurística de la pendiente")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=True)
            fig.tight_layout()

            if path is not None:
                fig.savefig(
                    path,
                    dpi=300,
                    bbox_inches="tight"
                )

            self.summary_figures["slope"] = fig

        if return_fig:
            return penals_sorted, min_costs, fig

        if plot:
            plt.close(fig)

        return penals_sorted, min_costs

    def slope_heuristic_regression(
        self,
        s_thresh,
        costs,
        penals,
        plot=True,
        path='slope_heuristic',
        given=True,
        geom=True,
        return_fig=False
    ):

        from Utils.tools import hallar_pendiente

        penals_sorted, min_costs, fig = self.slope_heuristic_fig(
            costs,
            penals,
            plot,
            path,
            s_thresh=s_thresh,
            given=given,
            geom=geom,
            return_fig=True
        )

        penals_arr = np.array(penals_sorted)
        costs_arr = np.array(min_costs)

        m = hallar_pendiente(
            penals_arr,
            costs_arr,
            s_thresh,
            given
        )

        if return_fig:
            return abs(m), fig

        return abs(m)

    def plot_distancias(self, suavizada=True, title=None, path=None, show=False):
        """
        Grafica la curva de distancias de Wasserstein (raw o suavizada).
        
        Parámetros:
        -----------
        suavizada : bool
            Si True, grafica la distancia suavizada; si False, la raw.
        title : str
            Título personalizado de la gráfica.
        path : str
            Ruta para guardar la figura.
        show : bool
            Si True, muestra la gráfica.
        
        Retorna:
        --------
        fig : matplotlib.figure.Figure
            La figura generada.
        """
        if self.last_distance_raw is None or self.last_distance_smooth is None:
            self.distancias()

        sns.set_theme(style="whitegrid", context="talk")

        if suavizada:
            dist = self.last_distance_smooth
            label = "Distancia de Wasserstein"
            color = "tab:purple"
            default_title = "Distancia de Wasserstein suavizada"
        else:
            dist = self.last_distance_raw
            label = "Distancia de Wasserstein"
            color = "magenta"
            default_title = "Distancia de Wasserstein sin suavizar"

        x = np.arange(len(dist), dtype=int) + self.window

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, dist, linewidth=2.4, color=color, label=label)
        ax.set_xlabel("Índice temporal")
        ax.set_ylabel("Distancia de Wasserstein")
        ax.set_title(default_title if title is None else title)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=True)
        fig.tight_layout()

        if path is not None:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig

    def plot_resumen(self, cps=None, pc_detectados=None, title=None, path=None, show=False):
        """
        Grafica un resumen general del algoritmo de detección de cambios.
        Incluye la serie temporal, puntos de cambio reales, puntos detectados
        y la curva de distancias de Wasserstein.
        
        Parámetros:
        -----------
        cps : array-like
            Puntos de cambio verdaderos (ground truth).
        pc_detectados : array-like
            Puntos de cambio detectados por el algoritmo.
        title : str
            Título personalizado.
        path : str
            Ruta para guardar la figura.
        show : bool
            Si True, muestra la gráfica.
        
        Retorna:
        --------
        fig : matplotlib.figure.Figure
            La figura generada.
        """
        if self.last_distance_raw is None or self.last_distance_smooth is None:
            self.distancias()

        sns.set_theme(style="whitegrid", context="talk")

        serie = np.asarray(self.Serie, dtype=np.float64)
        x_serie = np.arange(len(serie), dtype=int)
        x_dist = np.arange(len(self.last_distance_raw), dtype=int) + self.window

        cps_list = [] if cps is None else np.asarray(cps).ravel().tolist()
        pc_list = [] if pc_detectados is None else np.asarray(pc_detectados).ravel().tolist()

        fig, ax1 = plt.subplots(figsize=(15, 5.5))

        ax1.plot(
            x_serie,
            serie,
            color="0.75",
            linewidth=1.0,
            alpha=0.8,
            label="Serie de tiempo"
        )

        for i, cp in enumerate(cps_list):
            ax1.axvline(
                cp,
                color="black",
                linestyle=(0, (5, 4)),
                linewidth=2.0,
                alpha=0.95,
                label="Puntos de cambio originales" if i == 0 else None
            )

        for i, cp in enumerate(pc_list):
            ax1.axvline(
                cp,
                color="tab:purple",
                linestyle=(0, (5, 4)),
                linewidth=2.0,
                alpha=0.9,
                label="Puntos de cambio detectados" if i == 0 else None
            )

        ax2 = ax1.twinx()
        ax2.plot(
            x_dist,
            self.last_distance_smooth,
            color="tab:purple",
            linewidth=3.0,
            label="Distancia de Wasserstein"
        )
        '''ax2.plot(
            x_dist,
            self.last_distance_raw,
            color="magenta",
            linewidth=1.3,
            alpha=0.65,
            label="Distancia de Wasserstein sin suavizar"
        )'''

        ax1.set_xlabel("Índice temporal")
        ax1.set_ylabel("Serie de tiempo")
        ax2.set_ylabel("Distancia de Wasserstein")

        ax1.tick_params(axis="y", labelcolor="0.25")
        ax2.tick_params(axis="y", labelcolor="tab:purple")

        ax1.set_xlim(x_serie.min(), x_serie.max())
        ax2.set_xlim(x_serie.min(), x_serie.max())

        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles = handles1 + handles2
        labels = labels1 + labels2

        legend_map = {}
        for h, l in zip(handles, labels):
            if l not in legend_map and l != "_nolegend_":
                legend_map[l] = h

        ax1.legend(
            legend_map.values(),
            legend_map.keys(),
            loc="upper right",
            frameon=True
        )

        ax1.set_title("Resumen general del algoritmo de detección" if title is None else title)
        ax1.grid(True, alpha=0.2)
        fig.tight_layout()

        if path is not None:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig

    def build_summary_figure(
        self,
        cps_true=None,
        cps_detected=None,
        title='EmpiricalCPD Summary',
        path=None,
        show=False
    ):

        fig, axes = plt.subplots(
            3,
            1,
            figsize=(14, 10),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1, 1]}
        )

        ax0, ax1, ax2 = axes

        x = np.arange(len(self.Serie))

        ax0.plot(
            x,
            self.Serie,
            linewidth=1.2
        )

        if cps_true is not None:

            for cp in cps_true:

                ax0.axvline(
                    cp,
                    linestyle='--',
                    linewidth=1.5
                )

        if cps_detected is not None:

            for cp in cps_detected:

                ax0.axvline(
                    cp,
                    linestyle=':'
                )

        ax0.set_title(title)
        ax0.set_ylabel("Serie")

        if self.last_distance_raw is not None:

            ax1.plot(
                self.last_distance_raw,
                linewidth=1.2
            )

            ax1.set_ylabel("Wasserstein")

        if self.last_distance_smooth is not None:

            ax2.plot(
                self.last_distance_smooth,
                linewidth=1.5
            )

            ax2.set_ylabel("Smoothed")
            ax2.set_xlabel("Índice")

        for ax in axes:
            ax.grid(alpha=0.3)

        fig.tight_layout()

        if path is not None:

            fig.savefig(
                path,
                dpi=300,
                bbox_inches="tight"
            )

        self.summary_figures["summary"] = fig

        if show:
            plt.show()
        else:
            plt.close(fig)