import numpy as np
from Utils.detection import detect

_SHARED_SERIE = None

#Función utilizada para la paralelización
def init_worker(shared_array, length):
    global _SHARED_SERIE
    _SHARED_SERIE = np.frombuffer(shared_array, dtype=np.float64)[:length]

#Hace la evaluación de la función de coste para los parámetros (w, t) por fuerza bruta
def evaluate_params_worker(args):

    from cpd import CPD

    w, t, penal, lambda_p, class_defaults = args

    global _SHARED_SERIE
    serie = _SHARED_SERIE

    obj = CPD(
        serie,
        window=int(w),
        t=int(t),
        m=class_defaults.get("m", 3),
        medias=class_defaults.get("medias", True),
        sigma=class_defaults.get("sigma_filter", 2),
        k_gauss=class_defaults.get("k_gauss", True),
        n_perm=class_defaults.get("n_perm", 0),
    )

    distancias = obj.distancias()

    cps = detect(distancias, obj.window, alpha=0.05, thr=0)

    cp_full = np.concatenate(([0], cps, [len(serie)]))

    if not penal:
        cost = obj.total_cost(cp_full, penal=penal)
        return (cost, w, t, cps, distancias)
    else:
        total, penalty = obj.total_cost(cp_full, penal=penal)
        return (total, penalty, w, t, cps, distancias)
    

'''
Se aplica un metaheurístico híbrido que hace una búsqueda global con recocido simulado y ante cada mejora una búsqueda local
por first improvement. 
args contiene los parámetros:
(w0, t0): el punto inicial de búsqueda
min_w, max_w: Espacio de búsqueda para el tamaño de ventana
penal, lambda_p: Condicional y regularizador de la penalización
class_defaults: Parámetros en desuso del objeto CPD
max_iter: máxima cantidad de iteraciones para el metaheurístico
'''
def local_search_sa_worker(args):
    from cpd import CPD

    w0, t0, min_w, max_w, penal, lambda_p, class_defaults, max_iter = args

    global _SHARED_SERIE
    serie = _SHARED_SERIE

    rng = np.random.default_rng()

    cache = {}

    def evaluar(w, t):
        key = (w, t)
        if key in cache:
            return cache[key]

        obj = CPD(
            serie,
            window=int(w),
            t=int(t),
            m=class_defaults.get("m", 3),
            medias=class_defaults.get("medias", True),
            sigma=class_defaults.get("sigma_filter", 2),
            k_gauss=class_defaults.get("k_gauss", True),
        )

        distancias = obj.distancias()
        cps = detect(distancias, obj.window, alpha=0.05, thr=0)
        cp_full = np.concatenate(([0], cps, [len(serie)]))

        if penal:
            total, penalty = obj.total_cost(cp_full, penal=True)
            cost = total + lambda_p * penalty
        else:
            cost = obj.total_cost(cp_full, penal=False)

        cache[key] = (cost, cps, distancias)
        return cache[key]

    def hill_climb(w, t):

        improved = True
        best_cost, best_cps, best_dist = evaluar(w, t)

        steps = 0

        while improved:
            improved = False

            vecinos = [
                (w+1, t), (w-1, t),
                (w, t+1), (w, t-1),
                (w+1, t+1), (w-1, t-1)
            ]

            for w_new, t_new in vecinos:

                if not (min_w <= w_new <= max_w):
                    continue
                if not (1 <= t_new <= w_new//3):
                    continue

                cost, cps, dist = evaluar(w_new, t_new)

                if cost < best_cost:
                    w, t = w_new, t_new
                    best_cost = cost
                    best_cps = cps
                    best_dist = dist
                    improved = True
                    steps += 1
                    break 


        return w, t, best_cost, best_cps, best_dist


    w, t = w0, t0
    current_cost, current_cps, current_dist = evaluar(w, t)

    delta = abs(evaluar(min(w+1, max_w), t)[0] - current_cost)
    T = delta + 1e-6

    best_w, best_t = w, t
    best_cost = current_cost
    best_cps = current_cps
    best_dist = current_dist

    moves = [(-2,0),(2,0),(0,-1),(0,1),(1,1),(-1,-1),(1,-1),(-1,1)]


    for k in range(max_iter):

        T = T / (1 + k)  

        dw, dt = moves[rng.integers(len(moves))]
        w_new = int(np.clip(w + dw, min_w, max_w))
        t_new = int(np.clip(t + dt, 1, max(2, w_new//3)))

        new_cost, cps, dist = evaluar(w_new, t_new)

        delta = new_cost - current_cost

        if delta < 0 or rng.random() < np.exp(-delta / (T + 1e-9)):
            w, t = w_new, t_new
            current_cost = new_cost

            if new_cost < best_cost:

                w, t, hc_cost, hc_cps, hc_dist = hill_climb(w, t)

                current_cost = hc_cost

                best_w, best_t = w, t
                best_cost = hc_cost
                best_cps = hc_cps
                best_dist = hc_dist


    return best_cost, best_w, best_t, best_cps, best_dist