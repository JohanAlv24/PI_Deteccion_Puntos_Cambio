import numpy as np


def arima_serie(T, change_points, param_changes, p=1, q=1, seed=None):
    '''
    Genera una serie temporal sintética tipo ARIMA
    de longitud T, donde los parámetros del modelo pueden cambiar en ciertos puntos
    del tiempo.
    En los puntos de cambio, indicados por la lista change_points, los parámetros del modelo
    se actualizan según param_changes, permitiendo simular cambios de comportamiento 
    en la serie.

    Parámetros:
    - T: longitud de la serie
    - change_points: lista de puntos de cambio
    - param_changes: nuevos valores de parámetros en cada cambio
    - p, q: órdenes del modelo AR y MA
    - seed: semilla
    '''
    if seed is not None:
        np.random.seed(seed)

    params = {
        'phi': np.zeros(p),
        'theta': np.zeros(q),
        'sigma': 1.0,
        'c': 0.0
    }

    schedule = sorted(zip(change_points, param_changes), key=lambda x: x[0])
    change_idx = 0

    x = np.zeros(T)
    eps = np.zeros(T)

    for t in range(T):

        if change_idx < len(schedule) and t == schedule[change_idx][0]:
            for key, value in schedule[change_idx][1].items():
                params[key] = np.array(value) if isinstance(value, list) else value
            change_idx += 1

        eps[t] = np.random.normal(scale=params['sigma'])
        
        ar_term = 0.0
        for i in range(1, p + 1):
            if t - i >= 0:
                ar_term += params['phi'][i - 1] * x[t - i]

        ma_term = 0.0
        for j in range(1, q + 1):
            if t - j >= 0:
                ma_term += params['theta'][j - 1] * eps[t - j]

        x[t] = params['c'] + ar_term + ma_term + eps[t]

    return x
