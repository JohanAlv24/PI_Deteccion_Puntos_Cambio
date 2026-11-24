import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from scipy.linalg import fractional_matrix_power, logm
from numpy.lib.stride_tricks import sliding_window_view
from csaps import csaps
import multiprocessing as mp
from scipy.stats import norm
from Algoritmo_Gaussiano.workers import init_worker, evaluate_params_worker
from Utils.detection import detect


class CPD():
    '''
    X: serie de tiempo
    window: Tamaño de la ventana
    t: Retardo para aplicar el teorema de Takens
    m: Dimensión de los embeddings para aplicar el teorema de Takens
    medias: Condicional para usar la media en la distancia de Wasserstein
    smooth: Condicional para aplicar filtro gaussiano a la curva de distancias de Wasserstein
    n_perm: Parámetro en desuso para la prueba de permutación
    '''
    def __init__(self, X, window = 0, t = 0, m = 3,
                 medias = True, smooth = True, n_perm = 0, 
                 opt_delay=False, temporal_comp=False, max_comp=False, min_comp=False, 
                 max_bandwidth=False, prominence_ratio=0.05):
        
        self.Serie = np.asarray(X, dtype=np.float64)
        self.window = int(window)
        self.t = int(t)
        self.m = int(m)
        self.medias = bool(medias)
        self.smooth = bool(smooth)
        self.n_perm = int(n_perm)
        self.opt_delay = bool(opt_delay)
        self.temporal_comp = bool(temporal_comp)
        self.max_comp = bool(max_comp)
        self.min_comp = bool(min_comp)
        self.max_bandwidth = bool(max_bandwidth)
        self.prominence_ratio = float(prominence_ratio)
        self.vec_med = None
        self.Cov = None
        self.embeddings_list = []
        self._sorted_unique = None
        self._weights = None
        self.distancias_raw_ = None
        self.distancias_smooth_ = None
        self._distancias_x = None


   
        

    #La función Gaussian() genera la matriz de covarianzas y el vector de medias asociados a cada ventana de la serie de tiempo
    def Gaussian(self):

        serie = self.Serie
        w = self.window
        t = self.t
        m = self.m

        N_total = len(serie)
        n_windows = N_total - w + 1
     
        windows_view = sliding_window_view(serie, window_shape=w) 
        Covs = []
        Meds = []
        embeddings_firsts = []

        n_emb_per_window = w - (m - 1) * t


        base_idx = (np.arange(m) * t)[None, :]  
        starts = np.arange(n_emb_per_window)[:, None]  
        idx_matrix = starts + base_idx 


        for wi in range(n_windows):
            arr = windows_view[wi]  
            emb = arr[idx_matrix]
            embeddings_firsts.append(emb[0])

            comps = [emb]

            if self.temporal_comp:
                temporal = np.arange(
                    n_emb_per_window,
                    dtype=np.float64
                )

                temporal = (
                    temporal - temporal.mean()
                ) / temporal.std(ddof=1)

                comps.append(temporal[:, None])

            if self.min_comp:
                min_vals = np.min(emb, axis=1)[:, None]
                comps.append(min_vals)

            if self.max_comp:
                max_vals = np.max(emb, axis=1)[:, None]
                comps.append(max_vals)

            emb = np.hstack(comps)

            mu = emb.mean(axis=0)
        
            sum_xx = emb.T @ emb

            
            n_e = emb.shape[0]
            dim = emb.shape[1]
            if n_e > 1:
                cov = (sum_xx - n_e * np.outer(mu, mu)) / (n_e - 1)
            else:
                cov = np.zeros((dim, dim))

            Covs.append(cov)

            Meds.append(mu)

        self.Cov = np.array(Covs)
        self.vec_med = np.array(Meds)
        self.embeddings_list = np.array(embeddings_firsts)

    # ---------------------------

    '''
        distancias() calcula la distancia de Wasserstein entre ventanas consecutivas no traslapadas de manera
        vectorizada, por lo que devuelve la curva de distancias
    ''' 
    def distancias(self):
        self.Gaussian()
        d = []
        S1 = self.Cov[:len(self.Cov)-self.window]
        S2 = self.Cov[self.window:]
        m1 = self.vec_med[:len(self.Cov)-self.window]
        m2 = self.vec_med[self.window:]

        if self.medias:
            mean_sq = np.linalg.norm(m1 - m2, axis=1)**2
            cov_term = self.traces(S1, S2)
            d = np.sqrt(mean_sq + cov_term)
        else:
            d = np.sqrt(self.traces(S1, S2))
        
        d = np.asarray(d, dtype=np.float64)
        self.distancias_raw_ = d
        self._distancias_x = np.arange(len(d), dtype=int) + self.window

        if self.smooth:
            self.distancias_smooth_ = self.triangular_smooth()
            return self.distancias_smooth_

        self.distancias_smooth_ = d.copy()
        return d


    

    def plot_distancias(self, suavizada=True, title=None, path=None, show=False, save_fig=False):
        if self.distancias_raw_ is None or self.distancias_smooth_ is None:
            self.distancias()

        sns.set_theme(style="whitegrid", context="talk")

        if suavizada:
            dist = self.distancias_smooth_
            label = "Wasserstein Distance"
            color = "tab:purple"
            default_title = "Wasserstein Distance Smoothed"
        else:
            dist = self.distancias_raw_
            label = "Wasserstein Distance"
            color = "magenta"
            default_title = "Wasserstein Distance Unsmoothed"

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
        if self.distancias_raw_ is None or self.distancias_smooth_ is None:
            self.distancias()
    
        sns.set_theme(style="whitegrid", context="talk")

        serie = np.asarray(self.Serie, dtype=np.float64)
        x_serie = np.arange(len(serie), dtype=int)
        x_dist = np.arange(len(self.distancias_raw_), dtype=int) + self.window

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
        ax2.plot(
            x_dist,
            self.distancias_smooth_,
            color="tab:purple",
            linewidth=3.0,
            label="Wasserstein Distance"
        )
        '''ax2.plot(
            x_dist,
            self.distancias_raw_,
            color="magenta",
            linewidth=1.3,
            alpha=0.65,
            label="Wasserstein Distance Unsmoothed"
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

        ax1.set_title("Summary of the Change Point Detection Algorithm" if title is None else title)
        ax1.grid(True, alpha=0.2)
        fig.tight_layout()

        if path is not None and save_fig:
            fig.savefig(path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()

        return fig

    '''
    traces() recibe dos arreglos de matrices de covarianza (S1 y S2) y calcula, de forma vectorizada,
    el término de traza de la distancia de Wasserstein para cada par correspondiente de matrices.

    En particular, computa:
    tr(S1 + S2 - 2 * (S1^{1/2} S2 S1^{1/2})^{1/2})
    '''

    def traces(self, S1, S2, eps=1e-12):
        S1 = np.asarray(S1, dtype=np.float64)
        S2 = np.asarray(S2, dtype=np.float64)

        S1 = 0.5 * (
            S1 + np.swapaxes(S1, -1, -2)
        )

        S2 = 0.5 * (
            S2 + np.swapaxes(S2, -1, -2)
        )

        w1, v1 = np.linalg.eigh(S1)
        w1 = np.clip(w1, 0, None)

        sqrt_w1 = np.sqrt(w1)

        sqrt1 = (
            v1 * sqrt_w1[..., None, :]
        ) @ np.swapaxes(v1, -1, -2)

        # np.matmul realiza automáticamente el broadcasting:
        # S1 puede ser una matriz y S2 un lote de matrices.
        middle = sqrt1 @ S2 @ sqrt1

        middle = 0.5 * (
            middle
            + np.swapaxes(middle, -1, -2)
        )

        wm, vm = np.linalg.eigh(middle)
        wm = np.clip(wm, 0, None)

        sqrt_wm = np.sqrt(wm)

        sqrt_middle = (
            vm * sqrt_wm[..., None, :]
        ) @ np.swapaxes(vm, -1, -2)

        diff = S1 + S2 - 2.0 * sqrt_middle

        trace_values = np.trace(
            diff,
            axis1=-2,
            axis2=-1
        )

        return np.clip(trace_values, 0, None)



    def _stabilize_covariances(
        self,
        covariances,
        ridge
    ):
        covariances = np.asarray(
            covariances,
            dtype=np.float64
        )

        if covariances.ndim != 3:
            raise ValueError(
                "covariances debe tener forma (n, d, d). "
                f"Forma recibida: {covariances.shape}"
            )

        covariances = 0.5 * (
            covariances
            + covariances.transpose(0, 2, 1)
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariances
        )

        eigenvalues = np.maximum(
            eigenvalues,
            ridge
        )

        stabilized = (
            eigenvectors
            * eigenvalues[..., None, :]
        ) @ eigenvectors.transpose(0, 2, 1)

        return stabilized


    def triangular_smooth(self, distancias=None):

        if distancias is None:
            distancias = self.distancias_raw_
        
        y = np.asarray(distancias, dtype=np.float64)

        if self.max_bandwidth:
            window = int(np.min([int(self.window//2), int(np.sqrt(len(self.Serie)))]))
        else:
            window = int(self.window // 2)

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

    #Tangent() hace la proyección de las matrices de covarianza al plano tangente. Este método se usa para la etapa de clusterización
    def tangent(self, cov=None):
        if cov is not None:
            m_cov = cov
        else:
            m_cov = self.Cov

        centro = m_cov.mean(axis=0)
        
        centro_sqrt = fractional_matrix_power(centro, 0.5)
        centro_inv_sqrt = fractional_matrix_power(centro, -0.5)
        proyecciones = []
        for Sigma in m_cov:
            log = logm(centro_inv_sqrt @ Sigma @ centro_inv_sqrt)
            W = centro_sqrt @ log @ centro_sqrt
            proyecciones.append(W)

        proyecciones = np.array(proyecciones)
        indices = np.triu_indices(proyecciones.shape[1])
        
        proyecciones = np.array([
            matriz[indices]
            for matriz in proyecciones
        ])
    
        return np.array(proyecciones) 



    def segment_gaussian_parameters(
        self,
        change_points
    ):
  

        serie = np.asarray(
            self.Serie,
            dtype=np.float64
        )

        T = serie.size
        m = int(self.m)
        delay = int(self.t)

       
        cps = np.asarray(
                change_points,
                dtype=np.float64
            ).ravel()

        

        cps = np.round(cps).astype(int)

        
    
        internal_cps = cps[
            (cps > 0)
            & (cps < T)
        ]

        boundaries = np.unique(
            np.concatenate(
                (
                    np.array([0], dtype=int),
                    internal_cps,
                    np.array([T], dtype=int)
                )
            )
        )

        lengths = np.diff(
            boundaries
        )

        covariances = []

        if self.medias:
            means = []
        else:
            means = None

        n_embeddings_list = []

        embedding_offsets = (
            np.arange(m, dtype=int)
            * delay
        )

        for segment_index in range(
            len(boundaries) - 1
        ):
            start = int(
                boundaries[segment_index]
            )

            end = int(
                boundaries[segment_index + 1]
            )

            segment = serie[start:end]
            segment_length = end - start

            n_embeddings = (
                segment_length
                - (m - 1) * delay
            )

            starts = np.arange(
                n_embeddings,
                dtype=int
            )[:, None]

            index_matrix = (
                starts
                + embedding_offsets[None, :]
            )

            embeddings = segment[
                index_matrix
            ]

            components = [
                embeddings
            ]

       
            if self.temporal_comp:
                temporal = np.arange(
                    1,
                    n_embeddings + 1,
                    dtype=np.float64
                )[:, None]

                temporal_mean = np.mean(
                    temporal
                )

                temporal_std = np.std(
                    temporal,
                    ddof=0
                )

                if temporal_std > 0:
                    temporal = (
                        temporal
                        - temporal_mean
                    ) / temporal_std
                else:
                    temporal = np.zeros_like(
                        temporal
                    )

                components.append(
                    temporal
                )

            if self.min_comp:
                minimum_values = np.min(
                    embeddings,
                    axis=1
                )[:, None]

                components.append(
                    minimum_values
                )

            if self.max_comp:
                maximum_values = np.max(
                    embeddings,
                    axis=1
                )[:, None]

                components.append(
                    maximum_values
                )

            embedding_matrix = np.hstack(
                components
            )

          
            sample_mean = np.mean(
                embedding_matrix,
                axis=0
            )

            centered = (
                embedding_matrix
                - sample_mean
            )

            covariance = (
                centered.T
                @ centered
            ) / (n_embeddings - 1)

          
            covariance = 0.5 * (
                covariance
                + covariance.T
            )

            covariances.append(
                covariance
            )

            if self.medias:
                means.append(
                    sample_mean
                )

            n_embeddings_list.append(
                n_embeddings
            )

        covariances = np.stack(
            covariances,
            axis=0
        )

        if self.medias:
            means = np.stack(
                means,
                axis=0
            )

        return means, covariances

   
    def consecutive_segment_wasserstein(
        self,
        change_points
    ):

        segment_means, segment_covariances = (
            self.segment_gaussian_parameters(
                change_points
            )
        )

        segment_covariances = np.asarray(
            segment_covariances,
            dtype=np.float64
        )

        n_segments = segment_covariances.shape[0]

        # Si solamente existe un segmento, no hay pares
        # de segmentos consecutivos.
        if n_segments < 2:
            return 0.0

        covariance_terms = self.traces(
            segment_covariances[:-1],
            segment_covariances[1:]
        )

        if self.medias:
            segment_means = np.asarray(
                segment_means,
                dtype=np.float64
            )

            mean_differences = (
                segment_means[:-1]
                - segment_means[1:]
            )

            mean_terms = np.einsum(
                "ij,ij->i",
                mean_differences,
                mean_differences
            )
        else:
            mean_terms = np.zeros(
                n_segments - 1,
                dtype=np.float64
            )

        wasserstein_squared = np.maximum(
            mean_terms + covariance_terms,
            0.0
        )

        wasserstein_distances = np.sqrt(
            wasserstein_squared
        )

        return float(
            np.sum(wasserstein_distances)
        )


    
    '''
    La función total_cost recibe la lista de puntos de cambio (change_poins) y aplica la función de coste a cada segmento
    para hallar la función de coste total. En esta función también se añade retorna el factor de penalización pero es integrado 
    en el proceso de optimización
    '''

    def total_cost(
        self,
        change_points,
        penal=True
    ):
      

        N = len(self.Serie)

        if change_points is None:
            internal_change_points = np.array(
                [],
                dtype=int
            )
        else:
            internal_change_points = np.asarray(
                change_points
            ).ravel()

            internal_change_points = (
                internal_change_points[
                    np.isfinite(
                        internal_change_points
                    )
                ]
            ).astype(
                int,
                copy=False
            )

        # Se eliminan los extremos, duplicados y puntos
        # que no pertenecen al interior de la serie.
        internal_change_points = np.unique(
            internal_change_points[
                (internal_change_points > 0)
                & (internal_change_points < N)
            ]
        )

        n_change_points = (
            internal_change_points.size
        )

        n_segments = (
            n_change_points + 1
        )

        # -------------------------------------------------
        # Criterio con penalización externa
        # -------------------------------------------------

        consecutive_sum = (
                        self.consecutive_segment_wasserstein(
                            internal_change_points
                        )
                    )
        
        # Se maximiza la separación media entre
        # segmentos consecutivos. Como la búsqueda
        # minimiza, se cambia el signo.
        separation_cost = -(
                        consecutive_sum
                        / n_segments
                    )
        if penal:
            
            # La complejidad es el número de puntos de
            # cambio internos, no el número de segmentos
            # ni la cantidad de elementos de change_points
            # incluyendo los extremos.
            complexity = n_change_points

            return (
                float(separation_cost),
                int(complexity)
            )

        return float(separation_cost)

    '''
    Realiza la búsque exhaustiva de los parámetros t (retardo) y w (tamaño de ventana). 
    El parámetro m (dimensión de la matriz de covarianzas) se fija en 3.
    min_w y max_w definen el intervalo de búsqueda para la ventana, penal es un condicional
    para incorporar la penalización y lambda_p es el regularizador que por defecto es -1
    para llevarlo al mismo orden del coste (1/(log(T)*T))
    '''
    def opt_window_t(
        self,
        min_w=None,
        max_w=None,
        penal=False,
        lambda_p=-1,
        join=True,
        t_given=None
    ):
        T = len(self.Serie)

        if min_w is None:
            min_w = 9

        min_w = int(min_w)

        if min_w < 3:
            raise ValueError(
                "min_w debe ser mayor o igual que 3."
            )

        # Se exigen al menos tres valores en la curva
        # para que pueda existir un máximo interior.
        max_w_allowed = (T - 2) // 2

        if max_w is None:
            max_w = max_w_allowed
        else:
            max_w = min(
                int(max_w),
                max_w_allowed
            )

        if min_w > max_w:
            raise ValueError(
                "No existen ventanas factibles en el intervalo "
                f"[{min_w}, {max_w}]."
            )

        # lambda_p solo tiene efecto cuando los componentes
        # del criterio se unen dentro de esta función.
        use_joined_penalty = bool(
            penal and join
        )

        if use_joined_penalty:
            if lambda_p is None or lambda_p == -1:
                lambda_p = (
                    1
                    / (
                        3
                        * np.log(T)
                        * np.sqrt(T)
                    )
                )
            elif lambda_p < 0:
                raise ValueError(
                    "lambda_p debe ser no negativo o igual a -1."
                )

            effective_lambda = float(lambda_p)
        else:
            effective_lambda = None

        windows = np.arange(
            min_w,
            max_w + 1,
            dtype=int
        )

        class_defaults = {
            "m": int(self.m),
            "medias": bool(self.medias),
            "smooth": bool(self.smooth),
            "n_perm": int(self.n_perm),
            "temporal_comp": bool(self.temporal_comp),
            "max_comp": bool(self.max_comp),
            "min_comp": bool(self.min_comp),
            "max_bandwidth": bool(self.max_bandwidth),
            "prominence_ratio": float(self.prominence_ratio)
        }

        # Dimensión efectiva del embedding aumentado.
        dimension = (
            self.m
            + int(self.temporal_comp)
            + int(self.min_comp)
            + int(self.max_comp)
        )

        # Para que una covarianza d x d pueda ser de rango
        # completo se necesitan al menos d + 1 muestras.
        min_embeddings = dimension + 1

        if t_given is not None:
            t_given = int(t_given)

            if t_given < 1:
                raise ValueError(
                    "t_given debe ser mayor o igual que 1."
                )

        tasks = []

        for w in windows:

            if self.m == 1:
                delay_candidates = [
                    1 if t_given is None
                    else t_given
                ]

            elif t_given is not None:
                # Cuando el retardo es suministrado explícitamente,
                # se respeta siempre que produzca una configuración
                # estadísticamente factible.
                delay_candidates = [t_given]

            else:
                # Condición 1:
                # el span completo del embedding no puede ocupar
                # más de la mitad de la ventana.
                #
                # (m - 1)t <= w/2
                max_delay_by_span = (
                    w // (2 * (self.m - 1))
                )

                # Condición 2:
                # deben quedar al menos d + 1 embeddings.
                #
                # w - (m - 1)t >= d + 1
                max_delay_by_samples = (
                    (w - min_embeddings)
                    // (self.m - 1)
                )

                t_max = min(
                    max_delay_by_span,
                    max_delay_by_samples
                )

                if t_max < 1:
                    continue

                delay_candidates = range(
                    1,
                    t_max + 1
                )

            for t in delay_candidates:
                n_embeddings = (
                    w
                    - (self.m - 1) * t
                )

                if n_embeddings < min_embeddings:
                    continue

                tasks.append(
                    (
                        int(w),
                        int(t),
                        bool(penal),
                        class_defaults
                    )
                )

        if not tasks:
            raise ValueError(
                "La grilla no contiene configuraciones factibles. "
                "Aumente min_w, aumente max_w o reduzca t_given."
            )

        # Aproximación del trabajo de Gaussian():
        # número de ventanas por número de embeddings.
        # Las tareas más costosas se envían primero para reducir
        # el desbalance al final del procesamiento.
        tasks.sort(
            key=lambda task: (
                (T - task[0] + 1)
                * (
                    task[0]
                    - (self.m - 1) * task[1]
                )
            ),
            reverse=True
        )

        shared_array = mp.Array(
            "d",
            self.Serie,
            lock=False
        )

        length = len(self.Serie)

        f_costos = {}

        if penal:
            penalizaciones = {}

        if join:
            espacio = {}

        best_rank = None
        best_params = None

        n_jobs = min(
            max(1, mp.cpu_count() - 1),
            len(tasks)
        )

        # Las tareas son relativamente pesadas y heterogéneas.
        # Se utilizan bloques pequeños para conservar balance
        # sin enviar una tarea por mensaje necesariamente.
        chunksize = max(
            1,
            min(
                4,
                len(tasks)
                // max(1, 16 * n_jobs)
            )
        )

        with mp.Pool(
            processes=n_jobs,
            initializer=init_worker,
            initargs=(
                shared_array,
                length
            )
        ) as pool:

            results = pool.imap_unordered(
                evaluate_params_worker,
                tasks,
                chunksize=chunksize
            )

            for result in results:

                if penal:
                    (
                        total,
                        penalty,
                        w,
                        t
                    ) = result

                    f_costos[(w, t)] = total
                    penalizaciones[(w, t)] = penalty

                    if use_joined_penalty:
                        selection_cost = (
                            total
                            + effective_lambda * penalty
                        )
                    else:
                        # Cuando join=False no se utiliza una
                        # penalización provisional.
                        selection_cost = total

                else:
                    (
                        total,
                        w,
                        t
                    ) = result

                    f_costos[(w, t)] = total
                    selection_cost = total

                if join:
                    espacio[(w, t)] = selection_cost

                if not np.isfinite(selection_cost):
                    continue

                # Desempate determinista:
                # menor costo, luego menor ventana y menor retardo.
                candidate_rank = (
                    float(selection_cost),
                    int(w),
                    int(t)
                )

                if (
                    best_rank is None
                    or candidate_rank < best_rank
                ):
                    best_rank = candidate_rank
                    best_params = (
                        int(w),
                        int(t)
                    )

        if best_params is None:
            raise RuntimeError(
                "Todas las configuraciones produjeron "
                "costos no finitos."
            )

        self.window, self.t = best_params

        # Las curvas completas ya no se transfieren desde todos
        # los procesos. Se recalcula solamente la correspondiente
        # a la mejor configuración provisional.
        best_dist = self.distancias()

        best_cp = detect(
            best_dist,
            self.window,
            thr=0,
            min_separation= (self.m - 1) * self.t + (self.m+int(self.temporal_comp)) + 1,
            prominence_ratio=self.prominence_ratio
        )

        if penal:
            if join:
                return (
                    best_dist,
                    best_cp,
                    espacio,
                    f_costos,
                    penalizaciones
                )

            return (
                best_dist,
                best_cp,
                f_costos,
                penalizaciones
            )

        if join:
            return (
                best_dist,
                best_cp,
                espacio,
                f_costos
            )

        return (
            best_dist,
            best_cp,
            f_costos
        )

    '''
    Esta función es análoga a la anterior, pero por medio del metaheurístico que se le da en workers.
    Los parámetros cumplen la misma función añadiendo la máxima cantidad de iteraciones por búsqueda del
    metaheurístico.
    '''

    def slope_heuristic_fig(self, costs, penals, plot=True, path='slope_heuristic', s_thresh=0, given=True, geom=True, return_fig=False, save_fig=False):
        from Utils.tools import suavizar_media_movil, detectar_codo
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
        
        result = sorted(penal_min_cost.items(), key=lambda x: x[0])
        
        penals_sorted = [x[0] for x in result]
        min_costs = [x[1] for x in result]

        fig = None

        if plot:
            sns.set_theme(style="whitegrid", context="talk")
            fig, ax = plt.subplots(figsize=(10, 5))

            ax.plot(penals_sorted, min_costs, marker='o', linewidth=2.0, label='Minimum Cost')

            thresh = None
            idx_codo = None

            if given:
                thresh = s_thresh
            else:
                y_smooth = suavizar_media_movil(np.asarray(min_costs, dtype=np.float64))
                ax.plot(penals_sorted, y_smooth, linewidth=2.2, linestyle='--', label='Smoothed')
                idx_codo = detectar_codo(np.asarray(penals_sorted, dtype=np.float64), y_smooth, geom=geom)
                thresh = penals_sorted[idx_codo]
 

            if thresh is not None:
                ax.axvline(thresh, linestyle='--', linewidth=2.0, color='tab:red', label=f'Detected Elbow = {thresh:.2f}')
                idx_plot = int(np.argmin(np.abs(np.asarray(penals_sorted, dtype=np.float64) - thresh)))
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
                fig.savefig(path, dpi=300, bbox_inches="tight")

        if return_fig:
            return penals_sorted, min_costs, fig

        return penals_sorted, min_costs
    
    def slope_heuristic_regression(self, s_thresh, costs, penals, plot=True, path='slope_heuristic', given=True, geom=True, return_fig=False, save_fig=False):
        from Utils.tools import hallar_pendiente
        penals_sorted, min_costs, fig = self.slope_heuristic_fig(
            costs,
            penals,
            plot=plot,
            path=path,
            s_thresh=s_thresh,
            given=given,
            geom=geom,
            return_fig=True,
            save_fig=save_fig
        )
        penals_arr = np.array(penals_sorted)
        costs_arr = np.array(min_costs)
        
        m = hallar_pendiente(penals_arr, costs_arr, s_thresh, given, plot=False)
        if m > 0.01:
            m = 0
        if return_fig:
            return abs(m), fig

        return abs(m)
        




'''
    def heuristic_window_t(self, min_w=None, max_w=None, penal=False, lambda_p=-1, max_iter=50):

        T = len(self.Serie)

        if lambda_p==-1:
            lambda_p = 1/(3*np.log(T)*T**0.5)

        if not min_w:
            min_w = 9
        if not max_w:
            max_w = T // 2

        n_jobs = max(1, mp.cpu_count() - 1)

        class_defaults = {
            "m": self.m,
            "medias": self.medias,
            "smooth": self.smooth
        }

        shared_array = mp.Array('d', self.Serie, lock=False)
        length = len(self.Serie)

        rng = np.random.default_rng(123)

        n_starts = n_jobs

        tasks = []
        for _ in range(n_starts):
            w0 = rng.integers(min_w, max_w)
            t0 = rng.integers(1, w0//3)
            tasks.append((w0, t0, min_w, max_w, penal, lambda_p, class_defaults, max_iter))

        best_cost = np.inf
        best_params = None
        best_cp = None
        best_dist = None

        with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(shared_array, length)) as pool:
            results = pool.map(local_search_sa_worker, tasks)

        for cost, w, t, cps, dist in results:
            if cost < best_cost:
                best_cost = cost
                best_params = (w, t)
                best_cp = cps
                best_dist = dist

        if best_params is not None:
            self.window, self.t = best_params
            #Se acota la desviación estándar del filtro gaussiano en 12 para evitar suavizar en exceso la curva de distancias con grandes tamaños de ventana
            # smoothing sigma parameter removed

        return best_dist, best_cp


def nn_dimensiones(self, m):
        serie = self.Serie
        n = len(serie) - m*self.t
        A = np.zeros(n)
        embeddings_1 = np.array([serie[i : i + (m)*self.t : self.t] for i in range(n)])
        embeddings_2 = np.array([serie[i : i + (m+1)*self.t : self.t] for i in range(n)])


        distancias = np.max(
                        np.abs(
                            embeddings_1[:, None, :] -
                            embeddings_1[None, :, :]
                        ),
                        axis=2
                    )

        np.fill_diagonal(distancias, np.inf)

        vecinos = np.argmin(distancias, axis=1)
        dist_m = distancias[np.arange(n), vecinos]

        dist_m1 = np.max(
            np.abs(
                embeddings_2 -
                embeddings_2[vecinos]
            ),
            axis=1
        )

        return dist_m1 / dist_m



    def E_star(self, d):
        serie = np.asarray(self.Serie)
        t = self.t
        N = len(serie)

        n = N - d*t

        # Embeddings de dimensión d
        embeddings = np.array([
            serie[i : i + d*t : t]
            for i in range(n)
        ])

        distancias = np.max(
            np.abs(
                embeddings[:, None, :] -
                embeddings[None, :, :]
            ),
            axis=2
        )

        np.fill_diagonal(distancias, np.inf)

        vecinos = np.argmin(distancias, axis=1)

        nuevos = serie[d*t : d*t + n]

        diferencias = np.abs(
            nuevos - nuevos[vecinos]
        )

        return np.mean(diferencias)

    def best_dim(self, max_m=10):
        best_m = 1
        best_ratio = np.inf

        for m in range(1, max_m + 1):
            ratio = self.nn_dimensiones(m).mean()
            if ratio < best_ratio:
                best_ratio = ratio
                best_m = m

        return best_m


def bspline_smooth_cv(self, degree=3, n_s_values=20, random_state=42):
        y = self.distancias_raw_.copy()
        x = self._distancias_x.copy()

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
        
        
            def mutual_information(self, delay, n_points=None):

        x = np.asarray(self.Serie, dtype=float)

        if delay < 0 or delay >= len(x):
            raise ValueError("delay debe estar entre 0 y len(serie)-1")

        # Pares (x(t), x(t + delay))
        if delay == 0:
            x1 = x.copy()
            x2 = x.copy()
        else:
            x1 = x[:-delay]
            x2 = x[delay:]

        n_available = len(x1)

        if n_points is None:
            n_points = 2 ** int(np.floor(np.log2(n_available)))

        if n_points > n_available:
            raise ValueError("n_points no puede superar el número de pares disponibles.")

        if n_points < 4:
            raise ValueError("Muy pocos datos para calcular la información mutua.")

        if n_points & (n_points - 1):
            raise ValueError("n_points debe ser una potencia de 2.")

        x1 = x1[:n_points]
        x2 = x2[:n_points]

        N = n_points
        n_bits = int(np.log2(N))

      
        def rank_permutation(z):
            order = np.argsort(z, kind="mergesort")

            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(z))

            return ranks.astype(np.int64)

        s = rank_permutation(x1)
        q = rank_permutation(x2)

      
        def F(indices, depth):

            Nr = len(indices)

            if Nr <= 1:
                return 0.0

           
            if depth >= n_bits:
                return Nr * np.log2(Nr)

            remaining_bits = n_bits - depth

           
            if remaining_bits < 2:
                return Nr * np.log2(Nr)

     

            bit = n_bits - depth - 1

            s_bit = (s[indices] >> bit) & 1
            q_bit = (q[indices] >> bit) & 1

            child = 2 * s_bit + q_bit

            a = np.bincount(
                child,
                minlength=4
            ).astype(float)

            chi3 = (
                (4.0 / 3.0)
                * np.sum((a - Nr / 4.0) ** 2)
                / Nr
            )

         
            shift = n_bits - depth - 2

            s4 = (s[indices] >> shift) & 0b11
            q4 = (q[indices] >> shift) & 0b11

            cell16 = 4 * s4 + q4

            b = np.bincount(
                cell16,
                minlength=16
            ).astype(float)

            chi15 = (
                (16.0 / 15.0)
                * np.sum((b - Nr / 16.0) ** 2)
                / Nr
            )

            flat = (
                chi3 < 1.547
                and chi15 < 1.287
            )

            if flat:
                return Nr * np.log2(Nr)

   
            result = Nr * np.log2(4)

            for k in range(4):
                child_indices = indices[child == k]
                result += F(child_indices, depth + 1)

            return result

  
        indices = np.arange(N)

        F0 = F(indices, 0)

        I = F0 / N - np.log2(N)

        return I

    def optimal_delay(
        self,
        max_delay,
        n_segments=4,
        tolerance=0.05,
        min_agreement=1.0,
        plot=False
    ):
       
        import copy

        serie = np.asarray(self.Serie)

        delays = np.arange(1, max_delay + 1)

        
        full_MI = np.array([
            self.mutual_information(delay)
            for delay in delays
        ])

        segments = np.array_split(serie, n_segments)

        segment_MI = []

        for segment in segments:

            cpd_segment = copy.copy(self)
            cpd_segment.Serie = segment

            MI = np.array([
                cpd_segment.mutual_information(delay)
                for delay in delays
            ])

            segment_MI.append(MI)

        segment_MI = np.array(segment_MI)

        def normalize(curve):

            minimum = np.min(curve)
            maximum = np.max(curve)

            if np.isclose(maximum, minimum):
                return np.zeros_like(curve)

            return (curve - minimum) / (maximum - minimum)

        full_normalized = normalize(full_MI)

        segments_normalized = np.array([
            normalize(curve)
            for curve in segment_MI
        ])


        stable_low = segments_normalized <= tolerance

        agreement = stable_low.mean(axis=0)

        full_low = full_normalized <= tolerance


        candidates = (
            (agreement >= min_agreement)
            & full_low
        )

        candidate_delays = delays[candidates]


        if len(candidate_delays) > 0:

            best_delay = candidate_delays[0]

        else:

            stability_score = (
                segments_normalized.mean(axis=0)
                + full_normalized
            ) / 2

            best_delay = delays[
                np.argmin(stability_score)
            ]


        if plot:

            plt.figure(figsize=(11, 6))

            # Segmentos
            for i in range(n_segments):

                plt.plot(
                    delays,
                    segment_MI[i],
                    alpha=0.6,
                    linewidth=1.5,
                    label=f"Segmento {i + 1}"
                )

            # Serie completa
            plt.plot(
                delays,
                full_MI,
                linewidth=3,
                label="Serie completa"
            )

            # Retardo seleccionado
            plt.axvline(
                best_delay,
                linestyle="--",
                linewidth=2,
                label=f"Retardo seleccionado = {best_delay}"
            )

            plt.xlabel("Retardo")
            plt.ylabel("Información mutua (bits)")
            plt.title(
                "Selección estable del retardo mediante información mutua"
            )

            plt.xticks(delays)
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.show()


        return best_delay
        '''