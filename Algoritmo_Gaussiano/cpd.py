import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.linalg import fractional_matrix_power, logm
from numpy.lib.stride_tricks import sliding_window_view
from csaps import csaps
import multiprocessing as mp
from scipy.stats import norm
from Algoritmo_Gaussiano.workers import init_worker, evaluate_params_worker, local_search_sa_worker


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
                 medias = True, smooth = True, n_perm = 0):
        
        self.Serie = np.asarray(X, dtype=np.float64)
        self.window = int(window)
        self.t = int(t)
        self.m = int(m)
        self.medias = bool(medias)
        self.smooth = bool(smooth)
        self.n_perm = int(n_perm)

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

            mu = emb.mean(axis=0)

        
            sum_xx = emb.T @ emb
            n_e = emb.shape[0]
            if n_e > 1:
                cov = (sum_xx - n_e * np.outer(mu, mu)) / (n_e - 1)
            else:
                cov = np.zeros((m, m))

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
            x = np.arange(len(d))
            self.distancias_smooth_, _ = csaps(x, d, x)
            return self.distancias_smooth_

        self.distancias_smooth_ = d.copy()
        return d

    def plot_distancias(self, suavizada=True, title=None, path=None, show=False):
        if self.distancias_raw_ is None or self.distancias_smooth_ is None:
            self.distancias()

        sns.set_theme(style="whitegrid", context="talk")

        if suavizada:
            dist = self.distancias_smooth_
            label = "Distancia de Wasserstein"
            color = "tab:purple"
            default_title = "Distancia de Wasserstein suavizada"
        else:
            dist = self.distancias_raw_
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
                label="Puntos de Cambio Detectados" if i == 0 else None
            )

        ax2 = ax1.twinx()
        ax2.plot(
            x_dist,
            self.distancias_smooth_,
            color="tab:purple",
            linewidth=3.0,
            label="Distancia de Wasserstein"
        )
        '''ax2.plot(
            x_dist,
            self.distancias_raw_,
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

    '''
    traces() recibe dos arreglos de matrices de covarianza (S1 y S2) y calcula, de forma vectorizada,
    el término de traza de la distancia de Wasserstein para cada par correspondiente de matrices.

    En particular, computa:
    tr(S1 + S2 - 2 * (S1^{1/2} S2 S1^{1/2})^{1/2})
    '''
    def traces(self, S1, S2, eps=1e-12):
    
        S1 = 0.5 * (S1 + S1.transpose(0,2,1))
        S2 = 0.5 * (S2 + S2.transpose(0,2,1))

        w1, v1 = np.linalg.eigh(S1)
        w1 = np.clip(w1, 0, None) 
        sqrt_w1 = np.sqrt(w1)
        sqrt1 = (v1 * sqrt_w1[..., None, :]) @ v1.transpose(0,2,1)

        middle = sqrt1 @ S2 @ sqrt1
        middle = 0.5 * (middle + middle.transpose(0,2,1))

        wm, vm = np.linalg.eigh(middle)
        wm = np.clip(wm, 0, None) 
        sqrt_wm = np.sqrt(wm)
        sqrt_middle = (vm * sqrt_wm[..., None, :]) @ vm.transpose(0,2,1)

        diff = S1 + S2 - 2.0 * sqrt_middle
        traces = np.einsum('nii->n', diff)

        traces = np.clip(traces, 0, None)

        return traces

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

        entropy_term = F_hat * np.log(F_hat) + (1 - F_hat) * np.log(1 - F_hat)

        return -n * np.sum(entropy_term * weights)
    
  
    
    '''
    La función total_cost recibe la lista de puntos de cambio (change_poins) y aplica la función de coste a cada segmento
    para hallar la función de coste total. En esta función también se añade retorna el factor de penalización pero es integrado 
    en el proceso de optimización
    '''
    def total_cost(self, change_points, penal = True):
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
            T = len(self.Serie)
            #beta = np.log(T)**(2.1)/2
            penalty = len(change_points)

            return total, penalty
        return total
    '''
    Realiza la búsque exhaustiva de los parámetros t (retardo) y w (tamaño de ventana). 
    El parámetro m (dimensión de la matriz de covarianzas) se fija en 3.
    min_w y max_w definen el intervalo de búsqueda para la ventana, penal es un condicional
    para incorporar la penalización y lambda_p es el regularizador que por defecto es -1
    para llevarlo al mismo orden del coste (1/(log(T)*T))
    '''
    def opt_window_t(self, min_w = None,
                     max_w = None, penal = False, lambda_p = -1, join = True):

        T = len(self.Serie)
        if lambda_p == -1:
            lambda_p = 1/(3*np.log(T)*T**0.5)
        if not min_w:
            min_w = 9
        if not max_w:
            max_w = T // 2

        windows = np.arange(min_w, max_w + 1, dtype=int)

        # Prepare tasks: (w,t, penal, lambda_p, small config dict)
        tasks = []
        class_defaults = {
            "m": self.m,
            "medias": self.medias,
            "smooth": self.smooth
        }
        for w in windows:
            t_max = max(w // 3, round(np.log(w)))
            for t in range(1, t_max + 1):
                tasks.append((w, t, penal, lambda_p, class_defaults))
                

        # Create shared memory for self.Serie so workers don't copy the array each time
        shared_array = mp.Array('d', self.Serie, lock=False)
        length = len(self.Serie)

        best_cost = np.inf
        best_params = None
        best_cp = None
        best_dist = None
        
        if join:
            espacio = {}
        f_costos = {}
        penalizaciones = {}

        n_jobs = max(1, mp.cpu_count() - 1)  # leave one core free

        # Use Pool with initializer to set global _SHARED_SERIE in workers
        with mp.Pool(processes=n_jobs, initializer=init_worker, initargs=(shared_array, length)) as pool:
            for result in pool.imap_unordered(evaluate_params_worker, tasks, chunksize=64):
                if penal:
                    total, penalty, w, t, cps, distancias = result
                    f_costos[(w, t)] = total
                    penalizaciones[(w, t)] = penalty
                    cost = total + lambda_p * penalty

                else:
                    cost, w, t, cps, distancias = result

                if join:
                    espacio[(w, t)] = cost
                #print(f"w={w}, t={t}, cost={cost}")
                if cost < best_cost:
                    best_cost = cost
                    best_params = (w, t)
                    best_cp = cps
                    best_dist = distancias

        if best_params is not None:
            self.window, self.t = best_params
            # smoothing sigma parameter removed
        if penal:
            if join:
                return best_dist, best_cp, espacio, f_costos, penalizaciones
            return best_dist, best_cp, f_costos, penalizaciones
     
        return best_dist, best_cp, espacio
    
    '''
    Esta función es análoga a la anterior, pero por medio del metaheurístico que se le da en workers.
    Los parámetros cumplen la misma función añadiendo la máxima cantidad de iteraciones por búsqueda del
    metaheurístico.
    '''

    def slope_heuristic_fig(self, costs, penals, plot=True, path='slope_heuristic', s_thresh=0, given=True, geom=True, return_fig=False):
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

            ax.plot(penals_sorted, min_costs, marker='o', linewidth=2.0, label='Costo mínimo')

            thresh = None
            idx_codo = None

            if given:
                thresh = s_thresh
            else:
                y_smooth = suavizar_media_movil(np.asarray(min_costs, dtype=np.float64))
                ax.plot(penals_sorted, y_smooth, linewidth=2.2, linestyle='--', label='Suavizada')
                idx_codo = detectar_codo(np.asarray(penals_sorted, dtype=np.float64), y_smooth, geom=geom)
                thresh = penals_sorted[idx_codo]
 

            if thresh is not None:
                ax.axvline(thresh, linestyle='--', linewidth=2.0, color='tab:red', label=f'Codo detectado = {thresh:.2f}')
                idx_plot = int(np.argmin(np.abs(np.asarray(penals_sorted, dtype=np.float64) - thresh)))
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
                fig.savefig(path, dpi=300, bbox_inches="tight")

        if return_fig:
            return penals_sorted, min_costs, fig

        return penals_sorted, min_costs
    
    def slope_heuristic_regression(self, s_thresh, costs, penals, plot=True, path='slope_heuristic', given=True, geom=True, return_fig=False):
        from Utils.tools import hallar_pendiente
        penals_sorted, min_costs, fig = self.slope_heuristic_fig(
            costs,
            penals,
            plot=plot,
            path=path,
            s_thresh=s_thresh,
            given=given,
            geom=geom,
            return_fig=True
        )
        penals_arr = np.array(penals_sorted)
        costs_arr = np.array(min_costs)
        
        m = hallar_pendiente(penals_arr, costs_arr, s_thresh, given, plot=False)
        
        if return_fig:
            return abs(m), fig

        return abs(m)
        

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