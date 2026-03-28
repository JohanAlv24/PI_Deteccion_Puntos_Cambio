import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
from scipy.linalg import fractional_matrix_power, logm
from numpy.lib.stride_tricks import sliding_window_view
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
    sigma: Valor del parámetro del filtro gaussiano
    k_gauss: Condicional para aplicar filtro gaussiano a la curva de distancias de Wasserstein
    n_perm: Parámetro en desuso para la prueba de permutación
    '''
    def __init__(self, X, window = 0, t = 0, m = 3,
                 medias = True, sigma = 2, k_gauss = True, n_perm = 0):
        
        self.Serie = np.asarray(X, dtype=np.float64)
        self.window = int(window)
        self.t = int(t)
        self.m = int(m)
        self.medias = bool(medias)
        self.sigma_filter = int(sigma)
        self.k_gauss = bool(k_gauss)
        self.n_perm = int(n_perm)

        self.vec_med = None
        self.Cov = None
        self.embeddings_list = []
        self._sorted_unique = None
        self._weights = None

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
                
        if self.k_gauss:
            return gaussian_filter1d(np.array(d), sigma=self.sigma_filter)
        #return np.convolve(d, self.kernel_triangular(), mode='same')
        return d

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
        w1_clipped = np.clip(w1, a_min=eps, a_max=None)   
        sqrt_w1 = np.sqrt(w1_clipped)    
    

        sqrt1 = (v1 * sqrt_w1[..., None, :]) @ v1.transpose(0,2,1)
    
        middle = sqrt1 @ S2 @ sqrt1
        middle = 0.5 * (middle + middle.transpose(0,2,1)) 
    
        wm, vm = np.linalg.eigh(middle)
        wm_clipped = np.clip(wm, a_min=eps, a_max=None)
        sqrt_wm = np.sqrt(wm_clipped)
        sqrt_middle = (vm * sqrt_wm[..., None, :]) @ vm.transpose(0,2,1)
    
 
        diff = S1 + S2 - 2.0 * sqrt_middle
        traces = np.einsum('nii->n', diff) 
    
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
            beta = 3*np.log(T)
            penalty = 0.5*(beta * (len(change_points)) + np.sum(np.log(np.diff(change_points/T))))
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
                     max_w = None, penal = False, lambda_p = -1):

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
            "sigma_filter": self.sigma_filter,
            "k_gauss": self.k_gauss
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

                espacio[(w, t)] = cost
                #print(f"w={w}, t={t}, cost={cost}")
                if cost < best_cost:
                    best_cost = cost
                    best_params = (w, t)
                    best_cp = cps
                    best_dist = distancias

        if best_params is not None:
            self.window, self.t = best_params
            #Se acota la desviación estándar del filtro gaussiano en 12 para evitar suavizar en exceso la curva de distancias con grandes tamaños de ventana
            self.sigma_filter = min(math.ceil(math.sqrt(self.window)), 12)
        if penal:
            return best_dist, best_cp, espacio, f_costos, penalizaciones
     
        return best_dist, best_cp, espacio
    
    '''
    Esta función es análoga a la anterior, pero por medio del metaheurístico que se le da en workers.
    Los parámetros cumplen la misma función añadiendo la máxima cantidad de iteraciones por búsqueda del
    metaheurístico.
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
            "sigma_filter": self.sigma_filter,
            "k_gauss": self.k_gauss
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
            self.sigma_filter = min(int(np.sqrt(self.window)) + 1, 12)

        return best_dist, best_cp