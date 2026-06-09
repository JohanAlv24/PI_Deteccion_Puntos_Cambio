# Algoritmo CDP Paralelizado

Este proyecto implementa dos enfoques para la detección de puntos de cambio en series de tiempo:

- **Algoritmo gaussiano** (`Algoritmo_Gaussiano/`)
- **Algoritmo empírico** (`Algoritmo_Empiricas/`)

Además, incluye utilidades para:
- búsqueda de hiperparámetros,
- visualización de resultados,
- generación de boxplots comparativos,
- y experimentación con series sintéticas.

---

## 1. Requisitos generales

El proyecto trabaja con:

- `numpy`
- `scipy`
- `matplotlib`
- `seaborn`
- `pandas`

---

## 2. Uso de `best_params_sh`

La función `best_params_sh` centraliza la calibración de parámetros, la detección de puntos de cambio y la generación opcional de gráficas.

### Firma actual

```python
best_params_sh(
    dataset,
    cps,
    path='',
    name='',
    s_thresh=0,
    penal=True,
    lambda_p=0,
    modo='Gaussiano',
    given=True,
    geom=True,
    visualization=None
)
```

### Parámetros principales

- `dataset`: serie de tiempo a analizar.
- `cps`: puntos de cambio reales, usados para evaluación.
- `path`: ruta base para guardar figuras.
- `name`: sufijo de nombre para archivos.
- `s_thresh`: umbral usado en la heurística de codo cuando aplica.
- `penal`: activa o desactiva penalización.
- `lambda_p`: regularizador de la penalización.
- `modo`: Si `Gaussiano` usa el algoritmo gaussiano; `Empírico` usa el empírico.
- `given`: indica si el umbral del codo es dado o estimado.
- `geom`: selecciona el método geométrico para hallar el codo.
- `visualization`: diccionario de banderas para controlar gráficas.

### Claves disponibles en `visualization`

```python
visualization = {
    "summary": True,
    "raw_distance": True,
    "smooth_distance": True,
    "cost": True,
    "slope": True
}
```

### Qué hace cada bandera

- `"summary"`: genera la figura resumen del algoritmo.
- `"raw_distance"`: muestra la distancia de Wasserstein sin suavizar.
- `"smooth_distance"`: muestra la distancia suavizada.
- `"cost"`: genera mapas de calor o dispersión de costos, según el método.
- `"slope"`: activa la gráfica de slope heuristic.

### Ejemplo de uso

```python
from Utils.tools import best_params_sh

visualization = {
    "summary": True,
    "raw_distance": True,
    "smooth_distance": True,
    "cost": True,
    "slope": True
}

metrics_gauss = best_params_sh(
    dataset=serie,
    cps=changepoints_true,
    path='Graficas/',
    name='ejemplo_01',
    s_thresh=0,
    penal=True,
    lambda_p=0,
    modo='Gaussiano',
    given=True,
    geom=True,
    visualization=visualization
)
```

### Ejemplo con método empírico

```python
metrics_emp = best_params_sh(
    dataset=serie,
    cps=changepoints_true,
    path='Graficas/',
    name='ejemplo_empirico',
    s_thresh=0,
    penal=True,
    lambda_p=0,
    modo='Empírico',
    given=True,
    geom=True,
    visualization=visualization
)
```

---

## 3. Generación de boxplots comparativos

La función `boxplot_comp` permite comparar dos matrices o conjuntos de resultados mediante boxplots.

### Firma esperada

```python
boxplot_comp(M1, M2, columnas, title, path)
```

### Parámetros

- `M1`: primera matriz de datos.
- `M2`: segunda matriz de datos.
- `columnas`: nombres de columnas para la visualización.
- `title`: título general de la figura.
- `path`: ruta base donde se guarda la figura.

### Qué hace

La función organiza los datos en formato largo y genera boxplots comparando:

- **Gaussiana**
- **Empírica**

según las columnas seleccionadas. Es usado para comparar las métricas generadas por ambos métodos

### Ejemplo de uso

```python
from Utils.tools import boxplot_comp

boxplot_comp(
    M1=resultados_gauss,
    M2=resultados_emp,
    columnas = ['Mean Location Error', 'Precision', 'Recall', 'F1 Score', 'Accuracy', 'Falsos Positivos', 'Falsos Negativos', 'Verdaderos Positivos'],
    title='Comparación de métricas',
    path='Graficas/boxplot_comparacion.png'
)
```

---

## 4. Experimentación con series sintéticas

El archivo `Series_Prueba/experimentos.py` contiene funciones para generar datos sintéticos y evaluar el desempeño de los algoritmos.

### Función principal

```python
samples_200_arma(...)
```

### Propósito

Genera `N` series sintéticas tipo ARMA con cambios de régimen y outliers, y luego evalúa:

- algoritmo gaussiano,
- algoritmo empírico.

### Firma resumida

```python
samples_200_arma(
    seed,
    N=200,
    thr_dist=30,
    T=2000,
    phi=(0.3, 0.5),
    theta=(0.0, 0.0),
    random_phi=False,
    random_theta=False,
    min_seg=50,
    max_seg=150,
    base_mean=0.0,
    random_mean=True,
    mean_range=(-1.0, 1.0),
    base_std=0.5,
    random_std=True,
    std_range=(0.2, 1.2),
    outlier_interval=200,
    outlier_scale=6.0,
    s_thresh=0,
    seed2=None,
    tail=False,
    tail_list=None,
    given=True,
    geom=True,
    visualization=None
)
```

### Parámetros importantes

- `seed`: semilla global.
- `N`: número de experimentos.
- `thr_dist`: tolerancia usada en la evaluación.
- `T`: longitud de la serie sintética.
- `tail=True`: ejecuta únicamente los experimentos listados en `tail_list`.
- `visualization`: activa las figuras de resumen, distancias, costeo y slope heuristic.

### Ejemplo básico

```python
from Series_Prueba.experimentos import samples_200_arma

visualization = {
    "summary": True,
    "raw_distance": True,
    "smooth_distance": True,
    "cost": True,
    "slope": True
}

metricas_gauss, metricas_emp = samples_200_arma(
    seed=1234,
    N=5,
    T=2000,
    random_phi=True,
    random_theta=True,
    tail=False,
    given=True,
    geom=True,
    visualization=visualization
)
```

### Ejemplo usando `tail`

```python
metricas_gauss, metricas_emp = samples_200_arma(
    seed=1234,
    N=10,
    tail=True,
    tail_list=[2, 5, 8],
    visualization=visualization
)
```

En este caso solo se ejecutan los experimentos 2, 5 y 8.

## 5. Uso directo de los algoritmos (sin búsqueda exhaustiva)

En algunos escenarios puede ser conveniente fijar manualmente los hiperparámetros del algoritmo y omitir la búsqueda exhaustiva realizada por `best_params_sh`. Esto permite evaluar directamente una configuración específica.

### Ejemplo con el método empírico

```python
from Algoritmo_Empiricas.Empirical_CPD import EmpiricalCPD
from Utils.detection import detect
from Utils.metrics_sup import metrics

# Serie de tiempo
serie = ...

# Puntos de cambio reales
cps_reales = ...

algoritmo_emp = EmpiricalCPD(
    serie,
    window=100
)

distancias_emp = algoritmo_emp.distancias()

pc_detectados_emp = detect(
    distancias_emp,
    window=100
)

metricas_emp = metrics(
    cps_reales,
    pc_detectados_emp,
    threshold=50,
    T=len(serie)
)
```

### Ejemplo con el método gaussiano

```python
from Algoritmo_Gaussiano.cpd import CPD
from Utils.detection import detect
from Utils.metrics_sup import metrics

# Serie de tiempo
serie = ...

# Puntos de cambio reales
cps_reales = ...

algoritmo_gauss = CPD(
    serie,
    window=100,
    t=10
)

distancias_gauss = algoritmo_gauss.distancias()

pc_detectados_gauss = detect(
    distancias_gauss,
    window=100
)

metricas_gauss = metrics(
    cps_reales,
    pc_detectados_gauss,
    threshold=50,
    T=len(serie)
)
```
---



