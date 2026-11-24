import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from csaps import csaps
import multiprocessing as mp
import scipy.io as scio
from scipy.interpolate import UnivariateSpline
from Algoritmo_Empiricas.workers_empirical import init_worker_empirical, evaluate_window_worker
from Utils.detection import detect
import os

class EmpiricalCPD():
    '''
    X: serie de tiempo
    window: Tamaño de la ventana
    smooth: Condicional para aplicar filtro gaussiano a la curva de distancias de Wasserstein
    '''
    def __init__(self, X, window=0, smooth=True, max_bandwidth=False, prominence_ratio=0.05):

        self.Serie = np.asarray(X, dtype=np.float64)

        self.window = int(window)

        self.smooth = bool(smooth)

        self.max_bandwidth = bool(max_bandwidth)

        self._sorted_unique = None
        self._weights = None

        self.last_distance_raw = None
        self.last_distance_smooth = None
        self._distancias_x = None
        self.prominence_ratio = float(prominence_ratio)


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


    def bspline_smooth_cv(self, degree=3, n_s_values=20, random_state=42):
        y = self.last_distance_raw.copy()
        x = self._distance_x.copy()

        # Rango razonable para s
        variance = np.var(y)
        s_candidates = np.geomspace(
            max(1e-8, 0.1 * len(y) * variance),
            max(1e-7, 10 * len(y) * variance),
            n_s_values
        )

        best_s = None
        best_error = np.inf

        for s in s_candidates:
            errors = []

           

            spline = UnivariateSpline(
                    x,
                    y,
                    k=degree,
                    s=s
            )

            pred = spline(x)
            errors.append(np.mean((pred - y)**2))

            error = np.mean(errors)

            if error < best_error:
                best_error = error
                best_s = s

        spline = UnivariateSpline(x, y, k=degree, s=best_s)

        return spline(x)
    

    def triangular_smooth(self, distancias=None):

        if distancias is None:
            distancias = self.last_distance_raw

        
        y = np.asarray(distancias, dtype=np.float64)
        if self.max_bandwidth:
            window = min(int(self.window//2), int(np.sqrt(len(self.Serie))))

        else:
            window = int(self.window//2)

        offsets = np.arange(
            -window,
            window + 1
        )

        kernel = np.maximum(
            1.0 - np.abs(offsets) / window,
            0.0
        )

        kernel = kernel / kernel.sum()

        padded = np.pad(
            y,
            pad_width=window,
            mode="reflect"
        )

        smoothed = np.convolve(
            padded,
            kernel,
            mode="valid"
        )

        return smoothed

    #Calcula la distancia de Wasserstein entre ventanas consecutivas no traslapadas de la serie de tiempo y retorna la curva de distancia suavizada
    def distancias(self):

        w = self.window

        d = self.empirical_cpd(self.Serie, w)

        self.last_distance_raw = np.array(d)
        self._distance_x = np.arange(len(d), dtype=int) + self.window

        if self.smooth:

            #d_smooth, smooth_used = csaps(x, d, x)

            #self.last_distance_smooth = self.bspline_smooth_cv()
            self.last_distance_smooth = self.triangular_smooth()

            return self.last_distance_smooth

        self.last_distance_smooth = np.array(d)

        return d

    @staticmethod
    def _wasserstein_1d_sorted(x_sorted, y_sorted):
       
        x_sorted = np.asarray(x_sorted, dtype=np.float64).ravel()
        y_sorted = np.asarray(y_sorted, dtype=np.float64).ravel()

        n_x = x_sorted.size
        n_y = y_sorted.size

        if n_x == 0 or n_y == 0:
            return np.inf

        if n_x == n_y:
            return float(np.mean(np.abs(x_sorted - y_sorted)))

        support = np.sort(np.concatenate((x_sorted, y_sorted)))

        if support.size < 2:
            return 0.0

        intervals = np.diff(support)
        left_points = support[:-1]

        cdf_x = (
            np.searchsorted(x_sorted, left_points, side="right")
            / n_x
        )
        cdf_y = (
            np.searchsorted(y_sorted, left_points, side="right")
            / n_y
        )

        return float(
            np.sum(np.abs(cdf_x - cdf_y) * intervals)
        )

    def consecutive_segment_wasserstein(
        self,
        segment_distributions
    ):
        """Suma W1 entre distribuciones de segmentos consecutivos."""
        total = 0.0

        for left, right in zip(
            segment_distributions[:-1],
            segment_distributions[1:]
        ):
            total += self._wasserstein_1d_sorted(left, right)

        return float(total)

    def _segment_boundaries(self, change_points):
      
        N = len(self.Serie)

        if change_points is None:
            internal = np.empty(0, dtype=int)
        else:
            values = np.asarray(change_points).ravel()
            values = values[np.isfinite(values)].astype(int, copy=False)
            internal = np.unique(values[(values > 0) & (values < N)])

        return np.concatenate((
            np.array([0], dtype=int),
            internal,
            np.array([N], dtype=int)
        ))
    def segment_empirical_distributions(self, change_points):
      
        boundaries = self._segment_boundaries(change_points)

        segments = [
            np.sort(self.Serie[start:end])
            for start, end in zip(boundaries[:-1], boundaries[1:])
        ]

        lengths = np.diff(boundaries).astype(int, copy=False)

        return boundaries, segments, lengths


    '''
    La función total_cost recibe la lista de puntos de cambio (change_poins) y aplica la función de coste a cada segmento
    para hallar la función de coste total. En esta función también se añade retorna el factor de penalización pero es integrado 
    en el proceso de optimización
    '''
    def total_cost(self, change_points, penal=True):
        boundaries, segment_distributions, segment_lengths = (
            self.segment_empirical_distributions(change_points)
        )

        n_segments = len(segment_distributions)
        n_change_points = max(0, n_segments - 1)

        if penal:
            separation = self.consecutive_segment_wasserstein(
                segment_distributions
            )

            total = -separation / max(1, n_segments)

            return float(total), int(n_change_points)

        N = len(self.Serie)

        if n_segments <= 1 or N <= n_segments:
            return 0.0

        intra = self.variance_intrasegment(
            segment_distributions
        )
        inter = self.variance_intersegment(
            segment_distributions,
            segment_lengths
        )

        numerical_tolerance = (
            np.finfo(np.float64).eps
            * max(1.0, abs(inter), abs(intra))
        )

        if intra <= numerical_tolerance:
            return 0.0

        pseudo_f = (
            inter / (n_segments - 1)
        ) / (
            intra / (N - n_segments)
        )

        if not np.isfinite(pseudo_f):
            return 0.0

        return -float(pseudo_f)

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
        lambda_p=-1,
        join=True
    ):

        T = len(self.Serie)

        if min_w is None:
            min_w = 9

        min_w = int(min_w)

        if min_w < 3:
            raise ValueError(
                "min_w debe ser mayor o igual que 3."
            )

        max_w_allowed = (T - 2) // 2

        if max_w is None:
            max_w = max_w_allowed
        else:
            max_w = min(int(max_w), max_w_allowed)

        if min_w > max_w:
            raise ValueError(
                "No existen ventanas factibles en el intervalo "
                f"[{min_w}, {max_w}]."
            )

        use_joined_penalty = bool(penal and join)

        if use_joined_penalty:
            if lambda_p is None or lambda_p == -1:
                raise ValueError(
                    "Cuando penal=True y join=True debe suministrarse "
                    "un lambda_p no negativo. Si el regularizador se "
                    "calibra externamente, use join=False."
                )

            if lambda_p < 0:
                raise ValueError(
                    "lambda_p debe ser no negativo."
                )

            effective_lambda = float(lambda_p)
        else:
            effective_lambda = None

        windows = np.arange(min_w, max_w + 1, dtype=int)

        config = {
            "smooth": self.smooth,
            "max_bandwidth": self.max_bandwidth,
            "prominence_ratio": self.prominence_ratio
        }

        tasks = [
            (int(w), bool(penal), config)
            for w in windows
        ]

        tasks.sort(
            key=lambda task: (
                (T - task[0] + 1) * task[0]
            ),
            reverse=True
        )

        shared_array = mp.Array('d', self.Serie, lock=False)

        length = len(self.Serie)

        best_rank = None
        best_window = None

        if join:
            espacio = {}

        f_costos = {}
        if penal:
            penalizaciones = {}

        n_jobs = min(
            max(1, mp.cpu_count() - 1),
            len(tasks)
        )

        chunksize = max(
            1,
            min(
                8,
                len(tasks) // max(1, 8 * n_jobs)
            )
        )

        with mp.Pool(
            processes=n_jobs,
            initializer=init_worker_empirical,
            initargs=(shared_array, length)
        ) as pool:
            
            results = pool.imap_unordered(
                evaluate_window_worker,
                tasks,
                chunksize=chunksize
            )

            for result in results:

                if penal:

                    total, penalty, w = result

                    f_costos[w] = total
                    penalizaciones[w] = penalty

                    if use_joined_penalty:
                        selection_cost = (
                            total
                            + effective_lambda * penalty
                        )
                    else:
                        selection_cost = total

                else:

                    total, w = result

                    f_costos[w] = total
                    selection_cost = total

                if join:
                    espacio[w] = selection_cost

                if not np.isfinite(selection_cost):
                    continue

                candidate_rank = (
                    float(selection_cost),
                    int(w)
                )

                if best_rank is None or candidate_rank < best_rank:
                    best_rank = candidate_rank
                    best_window = int(w)

        if best_window is None:
            raise RuntimeError(
                "Todas las ventanas produjeron costos no finitos."
            )

        self.window = best_window

        best_dist = self.distancias()
        best_cp = detect(
            best_dist,
            self.window,
            thr=0,
            prominence_ratio = self.prominence_ratio
        )

        if penal:

            if join:
                return best_dist, best_cp, espacio, f_costos, penalizaciones

            return best_dist, best_cp, f_costos, penalizaciones

        if join:
            return best_dist, best_cp, espacio, f_costos

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
        return_fig=False,
        save_fig=False
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
                label='Minimum Cost',
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
                    label='Smoothed',
                )
                idx_codo = detectar_codo(np.asarray(penals_sorted, dtype=np.float64), y_smooth, geom=geom)
                thresh = penals_sorted[idx_codo]
  

            if thresh is not None:
                ax.axvline(
                    thresh,
                    linestyle='--',
                    linewidth=2.0,
                    color='tab:red',
                    label=f'Detected Elbow = {thresh:.2f}'
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

            ax.set_xlabel("Penalization")
            ax.set_ylabel("Minimum Cost")
            ax.set_title("Slope Heuristic")
            ax.grid(True, alpha=0.3)
            ax.legend(frameon=True)
            fig.tight_layout()

            if path is not None and save_fig:
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
        return_fig=False,
        save_fig=False
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
            return_fig=True,
            save_fig=save_fig
        )

        penals_arr = np.array(penals_sorted)
        costs_arr = np.array(min_costs)

        m = hallar_pendiente(
            penals_arr,
            costs_arr,
            s_thresh,
            given
        )

        if m > 0.01:
            m = 0

        if return_fig:
            return abs(m), fig

        return abs(m)

    def plot_distancias(self, suavizada=True, title=None, path=None, show=False, save_fig=False):
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
            label = "Wasserstein Distance (Smoothed)"
            color = "tab:purple"
            default_title = "Wasserstein Distance (Smoothed)"
        else:
            dist = self.last_distance_raw
            label = "Wasserstein Distance"
            color = "magenta"
            default_title = "Wasserstein Distance"

        x = np.arange(len(dist), dtype=int) + self.window

        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(x, dist, linewidth=2.4, color=color, label=label)
        ax.set_xlabel("Temporal Index")
        ax.set_ylabel("Wasserstein Distance")
        ax.set_title(default_title if title is None else title)
        ax.grid(True, alpha=0.2)
        ax.legend(frameon=True)
        fig.tight_layout()

        if path is not None and save_fig:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()


        return fig

    def plot_resumen(self, cps=None, pc_detectados=None, title=None, path=None, show=False, save_fig=False):
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
            label="Time Series"
        )

        for i, cp in enumerate(cps_list):
            ax1.axvline(
                cp,
                color="black",
                linestyle=(0, (5, 4)),
                linewidth=2.0,
                alpha=0.95,
                label="Original Change Points" if i == 0 else None
            )

        for i, cp in enumerate(pc_list):
            ax1.axvline(
                cp,
                color="tab:purple",
                linestyle=(0, (5, 4)),
                linewidth=2.0,
                alpha=0.9,
                label="Detected Change Points" if i == 0 else None
            )

        ax2 = ax1.twinx()
        # Eje x por defecto
        x_dist = np.arange(len(self.last_distance_raw), dtype=int) + self.window

        # Si las dimensiones no coinciden, reajustar el eje x
        # para que tenga la misma longitud que la curva a graficar.
        '''if len(x_dist) != len(self.last_distance_smooth):
           
            x_dist = np.arange(
                len(self.last_distance_smooth),
                dtype=int
            ) + self.window'''

        ax2.plot(
            x_dist,
            self.last_distance_smooth,
            color="tab:purple",
            linewidth=3.0,
            label="Wasserstein Distance (Smoothed)"
        )
        '''ax2.plot(
            x_dist,
            self.last_distance_raw,
            color="magenta",
            linewidth=1.3,
            alpha=0.65,
            label="Wasserstein Distance (Raw)"
        )'''

        ax1.set_xlabel("Temporal Index")
        ax1.set_ylabel("Time Series")
        ax2.set_ylabel("Wasserstein Distance")

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

        if path is not None and save_fig:
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
        show=False,
        save_fig=False
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

        if path is not None and save_fig:

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
