import numpy as np
import math
from scipy.ndimage import gaussian_filter1d
import multiprocessing as mp
from Algoritmo_Empiricas.workers_empirical import init_worker_empirical, evaluate_window_worker


class EmpiricalCPD():
    '''
    X: serie de tiempo
    window: Tamaño de la ventana
    sigma: Valor del parámetro del filtro gaussiano
    k_gauss: Condicional para aplicar filtro gaussiano a la curva de distancias de Wasserstein
    '''
    def __init__(self, X, window=0, sigma=2, k_gauss=True):

        self.Serie = np.asarray(X, dtype=np.float64)
        self.window = int(window)
        self.sigma_filter = int(sigma)
        self.k_gauss = bool(k_gauss)

        self._sorted_unique = None
        self._weights = None

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

        if self.k_gauss:
            return gaussian_filter1d(d, sigma=self.sigma_filter)

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

            T = len(self.Serie)

            #beta = np.log(T)**(2.1)/2
            beta = 3*np.log(T)
            penalty = 0.5*((beta * (len(change_points)) + np.sum(np.log(np.diff(change_points/T)))))

            
            return total, penalty

        return total

    '''
    Realiza la búsque exhaustiva del tamaño de ventana. 
    min_w y max_w definen el intervalo de búsqueda para la ventana, penal es un condicional
    para incorporar la penalización y lambda_p es el regularizador que por defecto es -1
    para llevarlo al mismo orden del coste (1/(log(T)*T))
    '''
    def opt_window(self,
                   min_w=None,
                   max_w=None,
                   penal=False,
                   lambda_p=-1):
        T = len(self.Serie)
        if lambda_p == -1:
            lambda_p = 1/(3*np.log(T)*T**0.5)

        if not min_w:
            min_w = 9

        if not max_w:
            max_w = T // 2

        windows = np.arange(min_w, max_w + 1, dtype=int)

        tasks = []

        config = {
            "sigma_filter": self.sigma_filter,
            "k_gauss": self.k_gauss
        }

        for w in windows:
            tasks.append((w, penal, lambda_p, config))


        shared_array = mp.Array('d', self.Serie, lock=False)
        length = len(self.Serie)

        best_cost = np.inf
        best_window = None
        best_cp = None
        best_dist = None

        espacio = {}
        f_costos = {}
        penalizaciones = {}

        n_jobs = max(1, mp.cpu_count() - 1)

        with mp.Pool(
            processes=n_jobs,
            initializer=init_worker_empirical,
            initargs=(shared_array, length)
        ) as pool:

            for result in pool.imap_unordered(
                    evaluate_window_worker,
                    tasks,
                    chunksize=64):

                if penal:
                    total, penalty,  w, cps, distancias  = result
                    f_costos[w] = total
                    penalizaciones[w] = penalty
                    cost = total + lambda_p * penalty
                else:
                    cost, w, cps, distancias = result

                espacio[w] = cost

                #print(f"w={w}, cost={cost}")

                if cost < best_cost:

                    best_cost = cost
                    best_window = w
                    best_cp = cps
                    best_dist = distancias

        if best_window is not None:

            self.window = best_window

            self.sigma_filter = min(
                math.ceil(math.sqrt(self.window)),
                12
            )
        if penal:
            return best_dist, best_cp, espacio, f_costos, penalizaciones
        return best_dist, best_cp, espacio