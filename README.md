# Detección gaussiana de puntos de cambio en series de tiempo univariadas

Este repositorio implementa un método fuera de línea para detectar múltiples
puntos de cambio en series de tiempo univariadas. El procedimiento representa
cada ventana mediante una distribución gaussiana construida sobre coordenadas
retardadas y compara ventanas adyacentes con la distancia de Wasserstein de
orden dos.

La implementación principal se encuentra en la clase `CPD`, incluida en
`Algoritmo_Gaussiano/cpd.py`.

## Descripción del método

Para una serie univariada

```text
x[0], x[1], ..., x[T-1].
```

Y cada ventana de longitud `w`, el método realiza los siguientes pasos:

1. Construye vectores de coordenadas retardadas de dimensión `m` y retardo
   `t`:

   ```text
   (x[j], x[j+t], ..., x[j+(m-1)t]).
   ```

2. Puede añadir una coordenada temporal normalizada mediante
   `temporal_comp=True`.
3. Estima la media y la matriz de covarianzas de los vectores contenidos en
   cada ventana.
4. Compara dos ventanas adyacentes y no solapadas mediante la distancia de
   Wasserstein entre distribuciones gaussianas:

   ```text
   W2² = ||mu_1 - mu_2||²
         + tr(Sigma_1 + Sigma_2
         - 2(Sigma_1^(1/2) Sigma_2 Sigma_1^(1/2))^(1/2)).
   ```

5. Suaviza el perfil de discrepancia con un filtro triangular cuando
   `smooth=True`.
6. Identifica máximos locales relevantes con la función `detect`.

Para una serie de longitud `T` y una ventana de tamaño `w`, el perfil contiene
`T - 2w + 1` valores. La función `detect` incorpora automáticamente el
desplazamiento de `w` observaciones y devuelve los puntos de cambio en los
índices de la serie original.

## Estructura principal

```text
Algoritmo_Gaussiano/
├── cpd.py                 # Clase CPD y calibración gaussiana
└── workers.py             # Evaluación paralela de configuraciones (w, t)

Utils/
├── detection.py           # Selección de máximos locales
├── metrics_sup.py         # Métricas de evaluación
└── tools.py               # Calibración y visualización
```

El repositorio también conserva una implementación empírica utilizada como
referencia experimental.

## Requisitos

```bash
pip install numpy scipy matplotlib seaborn pandas csaps
```

Los ejemplos deben ejecutarse desde la raíz del proyecto para que Python pueda
resolver los módulos `Algoritmo_Gaussiano` y `Utils`.

## Uso básico con una ventana fija

El siguiente ejemplo construye el perfil gaussiano, detecta los puntos de
cambio y calcula las métricas cuando se dispone de anotaciones de referencia.

```python
from Algoritmo_Gaussiano.cpd import CPD
from Utils.detection import detect
from Utils.metrics_sup import metrics


serie = ...
cps_reales = ...

w = 100
delta = 50

cpd_gauss = CPD(
    serie,
    window=w,
    t=1,
    m=3,
    medias=True,
    smooth=True,
    temporal_comp=True
)

distancias_gauss = cpd_gauss.distancias()

cps_detectados_gauss = detect(
    distancias_gauss,
    window=w,
    prominence_ratio=0.05
)

metricas_gauss, emparejamientos = metrics(
    cps_reales,
    cps_detectados_gauss,
    threshold=delta,
    T=len(serie)
)

print("Puntos detectados:", cps_detectados_gauss)
print("Métricas:", metricas_gauss)
```

La detección no utiliza `cps_reales`. Estas anotaciones intervienen únicamente
en la evaluación posterior.


Los parámetros relevantes para el flujo principal son:

| Parámetro | Descripción |
|---|---|
| `X` | Serie de tiempo univariada. |
| `window` | Número de observaciones en cada ventana. Debe permitir formar dos ventanas adyacentes dentro de la serie. |
| `t` | Retardo entre coordenadas consecutivas. En el uso directo se recomienda especificar un entero positivo. |
| `m` | Dimensión de la reconstrucción por coordenadas retardadas. |
| `medias` | Si es `True`, incluye la diferencia entre medias; si es `False`, utiliza únicamente el término de covarianzas. |
| `smooth` | Activa el filtro triangular sobre el perfil de discrepancia. |
| `temporal_comp` | Añade una coordenada temporal normalizada a cada vector retardado. |
| `min_comp` | Añade el mínimo de cada vector retardado. |
| `max_comp` | Añade el máximo de cada vector retardado. |
| `max_bandwidth` | Limita el semiancho del filtro triangular mediante la longitud de la serie. |
| `prominence_ratio` | Proporción del rango entre los cuantiles 0.05 y 0.95 utilizada para descartar máximos con contraste local muy reducido. |

## Detección de máximos

La función de detección es de la siguiente forma:

```python
detect(
    distancias,
    window,
    prominence_ratio=0.05,
    thr=0,
    min_separation=0
)
```

Primero identifica los máximos locales del perfil. Después conserva aquellos
cuya prominencia representa al menos `prominence_ratio` del intervalo
comprendido entre los cuantiles 0.05 y 0.95 de la curva. Este intervalo reduce
la influencia de valores extremos aislados.

El parámetro `min_separation` es opcional. Cuando es mayor que uno, los máximos
se examinan por orden decreciente de prominencia y solo se conservan
detecciones separadas por al menos esa cantidad de observaciones. En caso de
empate se prioriza la mayor altura y, posteriormente, el menor índice temporal.

Para reproducir en la detección final la separación mínima utilizada al
evaluar configuraciones dentro de `opt_window_t`, puede emplearse:

```python
min_separation = (
    (cpd_gauss.m - 1) * cpd_gauss.t
    + cpd_gauss.m
    + int(cpd_gauss.temporal_comp)
    + 1
)

cps_detectados_gauss = detect(
    distancias_gauss,
    window=cpd_gauss.window,
    prominence_ratio=cpd_gauss.prominence_ratio,
    min_separation=min_separation
)
```

## Métricas de evaluación

La función

```python
metrics(original_pc, detect_pc, threshold, T)
```

realiza un emparejamiento uno a uno dentro de la tolerancia `threshold` y
devuelve:

```python
metricas, emparejamientos = metrics(...)
```

El diccionario `metricas` contiene:

- `Mean Location Error`;
- `Precision`;
- `Recall`;
- `F1 Score`;
- `Accuracy`;
- `Falsos Positivos`;
- `Falsos Negativos`;
- `Verdaderos Positivos`.

## Visualización

Después de ejecutar `distancias()`, la misma instancia permite visualizar el
perfil o un resumen de la detección:

```python
cpd_gauss.plot_distancias(
    suavizada=True,
    show=True
)

cpd_gauss.plot_resumen(
    cps=cps_reales,
    pc_detectados=cps_detectados_gauss,
    show=True
)
```

Las curvas sin procesar y suavizada quedan disponibles en:

```python
cpd_gauss.distancias_raw_
cpd_gauss.distancias_smooth_
```

## Calibración de ventana y retardo

La función `best_params_sh`, definida en `Utils/tools.py`, permite realizar una
búsqueda exhaustiva de los parámetros gaussianos y aplicar la heurística de la
pendiente a la penalización por complejidad.

```python
from Utils.tools import best_params_sh


visualization = {
    "summary": False,
    "raw_distance": False,
    "smooth_distance": False,
    "cost": False,
    "slope": False
}

resultado_metricas, cps_detectados = best_params_sh(
    dataset=serie,
    cps=cps_reales,
    modo="Gaussiano",
    penal=True,
    min_w=30,
    max_w=200,
    t_given=1,
    mean_gauss=True,
    temporal_comp=True,
    smooth_distances=True,
    prominence_ratio=0.05,
    thr_dist=delta,
    visualization=visualization,
    return_cps_list=True,
    save_fig=False
)

metricas_optimizadas, emparejamientos = (
    resultado_metricas
)
```

Los límites `min_w` y `max_w` son inclusivos. Si `t_given` es un entero, el
retardo queda fijo y la búsqueda se realiza únicamente sobre la ventana. Si
`t_given=None`, se exploran los retardos estadísticamente factibles para cada
tamaño de ventana.

Los puntos de cambio de referencia suministrados en `cps` se utilizan para
calcular las métricas y para las visualizaciones solicitadas; no intervienen en
la función de costo empleada para seleccionar los parámetros.


## Flujo recomendado

```text
Serie univariada
        |
        v
Coordenadas retardadas por ventana
        |
        v
Media y matriz de covarianzas
        |
        v
Perfil de Wasserstein entre ventanas adyacentes
        |
        v
Suavizado triangular
        |
        v
Selección de máximos locales
        |
        v
Puntos de cambio detectados
```
