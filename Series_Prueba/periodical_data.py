import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def senal_sin_con_tendencia(amplitud, frecuencia, fase, pendiente, t):
    return amplitud * np.sin(2 * np.pi * frecuencia * t + fase) + pendiente * t


def _sample_parameters(N, amplitud_base, frecuencia_base, pendiente_base, fase_base, sigma_amp, sigma_freq, sigma_pend, sigma_fase, sigma_ruido):
    amplitudes = np.random.normal(loc=amplitud_base, scale=sigma_amp * amplitud_base, size=N)
    frecuencias = np.random.normal(loc=frecuencia_base, scale=sigma_freq, size=N)
    fases = np.random.normal(loc=fase_base, scale=sigma_fase, size=N)
    pendientes = np.random.normal(loc=pendiente_base, scale=sigma_pend, size=N)
    ruido_sigma = np.abs(np.random.normal(loc=sigma_ruido, scale=0.05 * sigma_ruido, size=N))
    frecuencias = np.clip(frecuencias, 0.001, None)
    amplitudes = np.clip(amplitudes, 0.01, None)
    return amplitudes, frecuencias, fases, pendientes, ruido_sigma


def _assign_subgrupos(N, n_subgrupos):
    base = np.repeat([f"S{i+1}" for i in range(n_subgrupos)], repeats=N // n_subgrupos)
    resto = N - len(base)
    for i in range(resto):
        base = np.append(base, f"S{(i % n_subgrupos) + 1}")
    indices_perm = np.random.permutation(N)
    subgrupos = base[indices_perm]
    return subgrupos, indices_perm


def _generate_cps(n_subgrupos, inicio_cp, fin_cp, separacion_minima, min_cps, max_cps, aleatorio=True, cps_principales=None, cambios_cp=None):
    if aleatorio:
        cp_counts = [np.random.randint(min_cps, max_cps + 1) for _ in range(n_subgrupos)]
        total_cps = sum(cp_counts)
        rejilla = list(range(inicio_cp, fin_cp, separacion_minima))
        if len(rejilla) < total_cps:
            paso = max(1, separacion_minima // 2)
            rejilla = list(range(inicio_cp, fin_cp, paso))
        if len(rejilla) < total_cps:
            raise RuntimeError("Ajusta 'd' o 'separacion_minima'; no hay suficientes posiciones para CPs.")
        seleccionadas = np.random.choice(rejilla, size=total_cps, replace=False)
        cps_principales = {}
        idx = 0
        for i in range(n_subgrupos):
            nombre = f"S{i+1}"
            ncp = cp_counts[i]
            cps_principales[nombre] = sorted(seleccionadas[idx: idx + ncp])
            idx += ncp
        tipos_cp = {}
        cambios_cp = {}
        for nombre in cps_principales:
            tipos = []
            cambios = []
            for _ in cps_principales[nombre]:
                tipo = np.random.choice(['amp', 'freq', 'fase', 'pend'])
                tipos.append(tipo)
                if tipo == 'amp':
                    factor = np.random.uniform(0.1, 2)
                    cambios.append(('amp', factor))
                elif tipo == 'freq':
                    delta = np.random.uniform(0.01, 0.05)
                    cambios.append(('freq', delta))
                elif tipo == 'fase':
                    delta_phi = np.random.uniform(-1.5, 1.5)
                    cambios.append(('fase', delta_phi))
                else:
                    delta_p = np.random.uniform(-0.2, 0.2)
                    cambios.append(('pend', delta_p))
            tipos_cp[nombre] = tipos
            cambios_cp[nombre] = cambios
        return cps_principales, tipos_cp, cambios_cp
    else:
        tipos_cp = {}
        for a in cambios_cp:
            tipos_cp[a] = list(map(lambda x: x[0], cambios_cp[a]))
        return cps_principales, tipos_cp, cambios_cp


def _build_series_for_subgroup(idx_series, amps, freqs, fases, pends, sigmas_r, cps, cambios, t, d):
    n_sub = len(idx_series)
    amps_t = np.repeat(amps[:, None], d, axis=1)
    freqs_t = np.repeat(freqs[:, None], d, axis=1)
    fases_t = np.repeat(fases[:, None], d, axis=1)
    pends_t = np.repeat(pends[:, None], d, axis=1)
    for cp_pos, cambio in zip(cps, cambios):
        tipo, val = cambio
        if tipo == 'amp':
            amps_t[:, cp_pos:] *= val
        elif tipo == 'freq':
            freqs_t[:, cp_pos:] += val
            freqs_t = np.maximum(freqs_t, 0.001)
        elif tipo == 'fase':
            fases_t[:, cp_pos:] += val
        elif tipo == 'pend':
            pends_t[:, cp_pos:] += val
    T = np.tile(t, (n_sub, 1))
    seno = amps_t * np.sin(2 * np.pi * freqs_t * T + fases_t)
    tendencia_lineal = pends_t * T
    ruido = np.random.normal(loc=0.0, scale=sigmas_r[:, None], size=(n_sub, d))
    señales_sub = seno + tendencia_lineal + ruido
    return señales_sub


def _plot_before_mpc(t, datos, subgrupos, n_subgrupos, senal_ref, m_pc):
    plt.figure(figsize=(10, 4.5))
    plt.title(f"Series de distintos subgrupos ANTES del primer CP global (t < {m_pc})")
    for i_sub in range(n_subgrupos):
        nombre = f"S{i_sub+1}"
        indices_subgrupo = np.where(subgrupos == nombre)[0]
        idx_series = np.random.choice(indices_subgrupo, size=min(3, len(indices_subgrupo)), replace=False)
        for idx_serie in idx_series:
            plt.plot(t[:m_pc], datos[idx_serie, :m_pc], label=f"{nombre} (idx={idx_serie})", alpha=0.7)
    plt.plot(t[:m_pc], senal_ref, linestyle=':', linewidth=1.2, label='señal base (referencia)')
    plt.xlabel("tiempo (t)")
    plt.ylabel("valor")
    plt.legend(loc='upper right', fontsize='small', ncol=2)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def _plot_examples_by_subgroup(t, datos, cps_principales, n_subgrupos, muestras_subgrupo, m_pc):
    for nombre in sorted(cps_principales.keys(), key=lambda x: int(x[1:])):
        indices = [m for m in range(datos.shape[0]) if subgrupos[m] == nombre][:muestras_subgrupo]
        plt.figure(figsize=(10, 3.6))
        plt.title(f"Ejemplos - {nombre} (CPs: {cps_principales[nombre]})")
        for idx in indices:
            plt.plot(t, datos[idx, :], alpha=0.9)
        for cp in cps_principales[nombre]:
            plt.axvline(cp, linestyle='--', linewidth=1.0)
        plt.axvline(m_pc, linestyle=':', linewidth=0.8, alpha=0.7)
        plt.xlabel("tiempo (t)")
        plt.ylabel("valor")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()


def _build_metadata(N, subgrupos, amplitudes, frecuencias, fases, pendientes, ruido_sigma, cps_principales, tipos_cp):
    metadatos = pd.DataFrame({
        "indice": np.arange(N),
        "subgrupo": subgrupos,
        "amplitud": amplitudes,
        "frecuencia": frecuencias,
        "fase": fases,
        "pendiente": pendientes,
        "ruido_sigma": ruido_sigma
    })
    metadatos["cps_principales"] = metadatos["subgrupo"].map(cps_principales)
    metadatos["tipos_cp"] = metadatos["subgrupo"].map(tipos_cp)
    return metadatos


def generar_series_pc(N, d, n_subgrupos, muestras_subgrupo, amplitud_base, 
                     frecuencia_base, pendiente_base, fase_base, sigma_amp, sigma_freq,
                     sigma_pend, sigma_ruido, inicio_cp=None, fin_cp=None, separacion_minima=None, 
                      min_cps=None, max_cps=None, cps_principales=None, cambios_cp=None, aleatorio=True, sigma_fase=0.5, graficar=True):
    t = np.arange(d)
    amplitudes_series, frecuencias_series, fases_series, pendientes_series, ruido_sigma_series = _sample_parameters(
        N, amplitud_base, frecuencia_base, pendiente_base, fase_base, sigma_amp, sigma_freq, sigma_pend, sigma_fase, sigma_ruido
    )
    subgrupos, indices_perm = _assign_subgrupos(N, n_subgrupos)
    amplitudes_series = amplitudes_series[indices_perm]
    frecuencias_series = frecuencias_series[indices_perm]
    fases_series = fases_series[indices_perm]
    pendientes_series = pendientes_series[indices_perm]
    ruido_sigma_series = ruido_sigma_series[indices_perm]
    cps_principales, tipos_cp, cambios_cp = _generate_cps(
        n_subgrupos, inicio_cp, fin_cp, separacion_minima, min_cps, max_cps, aleatorio, cps_principales, cambios_cp
    )
    datos = np.zeros((N, d))
    for i_sub in range(n_subgrupos):
        nombre = f"S{i_sub+1}"
        idx_series = np.where(subgrupos == nombre)[0]
        if len(idx_series) == 0:
            continue
        amps = amplitudes_series[idx_series].copy()
        freqs = frecuencias_series[idx_series].copy()
        fases = fases_series[idx_series].copy()
        pends = pendientes_series[idx_series].copy()
        sigmas_r = ruido_sigma_series[idx_series].copy()
        cps = cps_principales[nombre]
        cambios = cambios_cp[nombre]
        señales_sub = _build_series_for_subgroup(idx_series, amps, freqs, fases, pends, sigmas_r, cps, cambios, t, d)
        datos[idx_series, :] = señales_sub
   
    todos_cps = [cp for lista in cps_principales.values() for cp in lista]
    m_pc = min(todos_cps)
    senal_ref = senal_sin_con_tendencia(amplitud_base, frecuencia_base, fase_base, pendiente_base, t)[:m_pc]
    if graficar:
        _plot_before_mpc(t, datos, subgrupos, n_subgrupos, senal_ref, m_pc)
        for nombre in sorted(cps_principales.keys(), key=lambda x: int(x[1:])):
            indices = [m for m in range(N) if subgrupos[m] == nombre][:muestras_subgrupo]
            plt.figure(figsize=(10, 3.6))
            plt.title(f"Serie de Tiempo Puntos de Cambio Inducidos")
            for idx in indices:
                plt.plot(t, datos[idx, :], alpha=0.9)
            for cp in cps_principales[nombre]:
                plt.axvline(cp, linestyle='--', linewidth=1.0)
            plt.axvline(m_pc, color='k', linestyle=':', linewidth=0.8, alpha=0.7)
            plt.xlabel("tiempo (t)")
            plt.ylabel("valor")
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()

    return datos, subgrupos

def next_prob(i, tran_mat):
    return 1+np.argmax(np.random.multinomial(1, tran_mat[i-1]))
    
def serie_pc(tran_mat, exp, pc_params, min_w, n, seed=123):
    if seed!=None:
        np.random.seed(seed)
    cps = []
    cambios = []
    clusters_cps = [1]
    c_A, c_freq, c_fase = pc_params[0]

    t = int(min_w+np.random.exponential(exp[0]))
    next_i = next_prob(1, tran_mat)
    clusters_cps.append(next_i)
    for i in range(n):
        cps.append(t)
        next_A, next_freq, next_fase = pc_params[next_i-1]
        cambios.append(('amp', next_A/c_A))
        cambios.append(('freq', next_freq-c_freq))
        cambios.append(('fase', next_fase-c_fase))

        next_i = next_prob(next_i, tran_mat)
        c_A, c_freq, c_fase = next_A, next_freq, next_fase

        if i!=n-1:
            clusters_cps.append(next_i)

        t += int(min_w+np.random.exponential(exp[next_i-1]))
    
    return cps, cambios, clusters_cps

