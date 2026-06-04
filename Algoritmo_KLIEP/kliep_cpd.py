import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from csaps import csaps
from joblib import parallel_backend
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from roerich.change_point import ChangePointDetectionClassifier
from Utils.detection import detect
from Algoritmo_KLIEP.workers_kliep import init_worker_kliep, evaluate_window_worker_kliep


class KLIEP_CPD():
    '''
    X: serie de tiempo
    window: Tamaño de la ventana
    base_classifier: clasificador base de sklearn (default: QDA)
    metric: métrica de distancia ('klsym', 'pesym', 'jsd', 'mmd', 'fd')
    smooth: Condicional para aplicar filtro gaussiano a la curva de distancias
    '''
    def __init__(self, X, window=0, base_classifier=None, metric='klsym', periods=10, step=1, n_runs=1):

        self.Serie = np.asarray(X, dtype=np.float64)
        self.window = int(window)
        self.base_classifier = base_classifier if base_classifier is not None else QuadraticDiscriminantAnalysis(reg_param=5e-2)
        self.metric = metric
        self.periods = periods
        self.step = step
        self.n_runs = n_runs

        self._sorted_unique = None
        self._weights = None

        self.last_score = None
        self.last_distance_raw = None
        self.last_distance_smooth = None

        self.summary_figures = {
            "raw_distance": None,
            "smooth_distance": None,
            "cost_scatter": None,
            "summary": None,
            "slope": None
        }

    def distancias(self, return_cps=False):
        """
        Calcula la curva de distancias usando ChangePointDetectionClassifier
        y retorna la curva de distancia suavizada o raw
        """
        
        # Crear instancia del clasificador con parámetros actuales
        cpd_classifier = ChangePointDetectionClassifier(
            base_classifier=self.base_classifier,
            metric=self.metric,
            window_size=self.window,
            periods=self.periods,
            step=self.step,
            n_runs=self.n_runs
        )

        # Predecir cambios (score es la curva de distancias)
        with parallel_backend("threading", n_jobs=1):
            score, cps_pred = cpd_classifier.predict(self.Serie)

        self.last_score = score
        self.last_distance_raw = np.array(score, dtype=np.float64)

        d = self.last_distance_raw.copy()

        
        if return_cps:
            return cps_pred
        return d

    def mle(self):
        """
        Calcula máxima verosimilitud basada en la distribución empírica
        """

        sorted_unique = np.sort(np.unique(self.Serie))
        T = len(sorted_unique)
        u = np.arange(1, T + 1)
        weights = 1.0 / ((u - 0.5) * (T - u + 0.5))

        self._sorted_unique = sorted_unique
        self._weights = weights

        return sorted_unique, weights

    def segment_cost_mle(self, start, end, sorted_unique, weights):
        """
        Calcula el costo de un segmento de la serie temporal bajo máxima verosimilitud
        basado en la entropía de una distribución empírica.
        """


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

    def total_cost(self, change_points, penal=True):
        """
        Calcula el costo total de la segmentación usando los puntos de cambio dados.
        change_points debe incluir 0 y len(Serie) al inicio y fin.
        """
        sorted_unique, weights = self.mle()

        total = 0.0

        for i in range(len(change_points) - 1):
            start = change_points[i]
            end = change_points[i + 1]
            total += self.segment_cost_mle(start, end, sorted_unique, weights)

        if penal:
            # Penalización MBIC: -0.5 * (número de parámetros) * log(T)
            penalty = len(change_points)
            return total, penalty

        return total

    def opt_window(self, min_w=None, max_w=None, penal=False, lambda_p=0, join=True):
        """
        Realiza búsqueda exhaustiva del tamaño de ventana de forma paralelizada.
        min_w y max_w definen el intervalo de búsqueda.
        penal: incluir penalización
        lambda_p: regularizador de penalización
        join: si False, retorna estadísticas de costos por ventana
        """
        T = len(self.Serie)

        if not min_w:
            min_w = int(min(9, T // 200))

        if not max_w:
            max_w = int(T // 10)

        print("\n" + "=" * 70)
        print(" KLIEP_CPD :: Exhaustive Window Optimization")
        print("=" * 70)
        print(f" Serie length                : {T}")
        print(f" Window search interval      : [{min_w}, {max_w}]")
        print(f" Metric                      : {self.metric}")
        print(f" Penalized optimization      : {penal}")
        print("=" * 70)

        windows = np.arange(min_w, max_w + 1, dtype=int)

        tasks = []

        config = {
            "metric": self.metric,
            "base_classifier": self.base_classifier,
        }

        for w in windows:
            tasks.append((w, penal, lambda_p, config))

        shared_array = mp.Array('d', self.Serie, lock=False)
        length = len(self.Serie)

        best_cost = np.inf
        best_window = None
        best_cp = None
        best_dist = None

        f_costos = {}
        penalizaciones = {}

        n_jobs = max(1, mp.cpu_count() - 1)

        print(f" Parallel workers            : {n_jobs}")
        print(f" Total candidate windows     : {len(tasks)}")
        print("-" * 70)

        with mp.Pool(
            processes=n_jobs,
            initializer=init_worker_kliep,
            initargs=(shared_array, length)
        ) as pool:

            for idx, result in enumerate(
                pool.imap_unordered(
                    evaluate_window_worker_kliep,
                    tasks,
                    chunksize=64
                )
            ):
                if penal:
                    cost, penalty, w, change_points, distancias = result
                    total_with_penalty = cost + lambda_p * penalty
                    f_costos[w] = cost
                    penalizaciones[w] = penalty

                    if total_with_penalty < best_cost:
                        best_cost = total_with_penalty
                        best_window = w
                        best_cp = change_points
                        best_dist = distancias

                    '''print(
                        f" Window: {w:3d} | Cost: {cost:12.6f} | "
                        f"Penalty: {penalty:10.6f} | Total: {total_with_penalty:12.6f}"
                    )'''
                else:
                    cost, w, change_points, distancias = result
                    f_costos[w] = cost

                    if cost < best_cost:
                        best_cost = cost
                        best_window = w
                        best_cp = change_points
                        best_dist = distancias

                    '''print(f" Window: {w:3d} | Cost: {cost:12.6f}")'''

        if best_window is not None:
            self.window = best_window
            self.last_distance_raw = best_dist if best_dist is not None else self.distancias()

        print("-" * 70)
        print(" Optimización Finalizada")
        print(f" ventana óptima              : {self.window}")
        print(f" Mejor valor función objetivo        : {best_cost:.6f}")
        print("=" * 70 + "\n")

        if penal:
            return best_dist, best_cp, f_costos, penalizaciones
        else:
            return best_dist, best_cp, f_costos

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
        """
        if self.last_distance_raw is None or self.last_distance_smooth is None:
            print("Debe ejecutar distancias() primero")
            return None

        sns.set_theme(style="whitegrid", context="talk")

        if suavizada:
            dist = self.last_distance_smooth
            color = "tab:purple"
            label = "Distancia suavizada"
            default_title = "Curva de distancias suavizada - KLIEP_CPD"
        else:
            dist = self.last_distance_raw
            color = "tab:blue"
            label = "Distancia raw"
            default_title = "Curva de distancias raw - KLIEP_CPD"

        x = np.arange(len(dist), dtype=int) + self.window

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, dist, linewidth=2.4, color=color, label=label)
        ax.set_xlabel("Índice temporal")
        ax.set_ylabel("Distancia (métrica: {})".format(self.metric))
        ax.set_title(default_title if title is None else title)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=True)
        fig.tight_layout()

        if path is not None:
            try:
                fig.savefig(path + '.png', dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f"Error guardando figura: {e}")

        if show:
            try:
                plt.show()
            except Exception:
                pass

        return fig

    def plot_resumen(self, cps=None, pc_detectados=None, title=None, path=None, show=False):
        """
        Grafica un resumen completo con serie de tiempo, puntos de cambio y curva de distancias
        """
        if self.last_distance_raw is None or self.last_distance_smooth is None:
            print("Debe ejecutar distancias() primero")
            return None

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
            ax1.axvline(cp, color="red", linestyle="--", linewidth=2.0, alpha=0.8)
            if i == 0:
                ax1.lines[-1].set_label("Puntos de cambio verdaderos")

        for i, cp in enumerate(pc_list):
            ax1.axvline(cp, color="green", linestyle=":", linewidth=2.5, alpha=0.7)
            if i == 0:
                ax1.lines[-1].set_label("Puntos de cambio detectados")

        ax2 = ax1.twinx()
        ax2.plot(
            x_dist,
            self.last_distance_smooth,
            color="tab:purple",
            linewidth=3.0,
            label="Distancia ({})".format(self.metric)
        )

        ax1.set_xlabel("Índice temporal")
        ax1.set_ylabel("Serie de tiempo")
        ax2.set_ylabel("Distancia")

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
            if l not in legend_map:
                legend_map[l] = h

        ax1.legend(
            legend_map.values(),
            legend_map.keys(),
            loc="upper right",
            frameon=True
        )

        ax1.set_title("Resumen - KLIEP_CPD" if title is None else title)
        ax1.grid(True, alpha=0.2)
        fig.tight_layout()

        if path is not None:
            try:
                fig.savefig(path + '.png', dpi=150, bbox_inches='tight')
            except Exception as e:
                print(f"Error guardando figura: {e}")

        if show:
            try:
                plt.show()
            except Exception:
                pass

        return fig
