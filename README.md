# Clasificación de cobertura forestal con aprendizaje supervisado — Covertype

Notebook: [`aprendizaje_supervisado.ipynb`](aprendizaje_supervisado.ipynb), ejecutado y con todas las
salidas guardadas (tablas, figuras, matriz de confusión y reporte por clase).

---

## Resumen del problema

Se predice el **tipo de cobertura forestal** (7 clases) de una parcela de 30×30 m a partir de 54
variables cartográficas: altitud, orientación, pendiente, distancias al agua / a carreteras / a puntos
de incendio, índices de sombreado, área silvestre y tipo de suelo.

Es un problema de **clasificación multiclase con desbalance marcado** (ratio 103:1 entre la clase
mayoritaria y la minoritaria), y por eso el criterio de selección de modelo es el **F1 macro** y no la
accuracy.

---

## Datos

**Covertype** (UCI Machine Learning Repository, Blackard & Dean, 1998): mediciones del *U.S. Forest
Service* y del *USGS* en el *Roosevelt National Forest* (Colorado).

- **581.012 registros**, 54 variables, 7 clases de cobertura.
- Se obtiene con `sklearn.datasets.fetch_covtype`, **sin autenticación** (~11 MB comprimidos).

| Clase | Registros | % |
|---|---|---|
| Spruce/Fir (Pícea/Abeto) | 211.840 | 36,46 % |
| Lodgepole Pine (Pino contorta) | 283.301 | 48,76 % |
| Ponderosa Pine (Pino ponderosa) | 35.754 | 6,15 % |
| Cottonwood/Willow (Álamo/Sauce) | 2.747 | 0,47 % |
| Aspen (Álamo temblón) | 9.493 | 1,63 % |
| Douglas-fir (Abeto de Douglas) | 17.367 | 2,99 % |
| Krummholz | 20.510 | 3,53 % |

---

## Preparación

| Paso | Resultado |
|---|---|
| Nulos | 0 en las 581.012 filas |
| Duplicados | 0 |
| Categóricas | `Wilderness_Area` (4) y `Soil_Type` (40) ya vienen en one-hot y son coherentes (un 1 por fila): no hace falta `OneHotEncoder` |
| Desbalance | 103:1 → reparto estratificado, métricas macro y una variante con `class_weight="balanced_subsample"` |
| Escalado | Solo para la Regresión Logística, con `StandardScaler` **dentro de un Pipeline** ajustado únicamente con train (evita *data leakage*). Los modelos de árbol son invariantes a la escala |
| Reparto | 80 % entrenamiento / 20 % prueba (116.203 registros), estratificado por clase |

---

## Modelos comparados

Ninguno es Regresión Lineal. Se comparan tres baselines/alternativas y se elige el mejor por **F1
macro**.

| Modelo | Accuracy | F1 macro | Recall macro | Precision macro |
|---|---|---|---|---|
| Dummy (clase mayoritaria) | 0,4876 | 0,0937 | 0,1429 | 0,0697 |
| Regresión Logística (escalada) | 0,7235 | 0,5304 | 0,5095 | 0,5913 |
| HistGradientBoosting | 0,8850 | 0,8384 | 0,8439 | 0,8343 |
| **Random Forest** ← elegido | **0,9548** | **0,9250** | 0,9071 | 0,9459 |
| Random Forest (balanced) | 0,9553 | 0,9246 | 0,9069 | 0,9449 |

**Por qué Random Forest:** las fronteras entre clases son no lineales e involucran interacciones (la
misma altitud implica especies distintas según el área silvestre y el tipo de suelo); los árboles las
modelan de forma natural, no necesitan escalado, son robustos a outliers y a distribuciones
asimétricas, el *bagging* reduce la varianza, con 465k filas pueden capturar patrones geográficos
locales muy finos, y además dan importancia de variables.

El modelo final se elige automáticamente por F1 macro (0,9250 frente a 0,9246 de la variante
balanceada), por lo que se queda el **Random Forest sin ponderar**.

---

## Resultados del modelo final

| Métrica | Macro | Ponderado |
|---|---|---|
| Accuracy | 0,9548 | 0,9548 |
| Precision | 0,9459 | 0,9549 |
| Recall | 0,9071 | 0,9548 |
| F1-Score | **0,9250** | 0,9545 |

- Errores en el conjunto de prueba: **5.257 de 116.203** (4,5 %).
- La mayor fuente de error es **Pícea/Abeto → Pino contorta** (2.314 casos, el 44 % de todos los
  errores): son las dos clases mayoritarias y comparten rango de altitud.
- Recall en la clase minoritaria *Álamo temblón*: 0,7762 (frente a 0,0058 de la Regresión Logística).
- Accuracy en train 1,0000 frente a 0,9548 en test: sobreajuste leve, propio de un RF sin poda.
- **`Elevation` es la variable más importante** por impureza (MDI) y por permutación.

---

## Cómo ejecutarlo

**Google Colab.** Sube el `.ipynb` y pulsa *Ejecutar todo*: la primera celda instala lo que falte y la
descarga del dataset es automática.

**Local.**

```bash
python3 -m venv .venv            # o: uv venv .venv --python 3.12
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace aprendizaje_supervisado.ipynb
```

El entrenamiento de los cuatro modelos sobre ~465k registros tarda unos minutos en CPU.

---

## Estructura del repositorio

```
.
├── aprendizaje_supervisado.ipynb   # notebook ejecutado
├── requirements.txt
└── README.md
```

---

## Advertencias

1. **Fuga de información geográfica.** Las parcelas vecinas son casi idénticas y el reparto aleatorio
   las mezcla entre train y test; las distancias absolutas funcionan como coordenadas, así que el
   modelo puede estar memorizando zonas. El rendimiento en un bosque distinto sería menor.
2. **Desbalance.** Las clases 4 (0,5 %) y 5 (1,6 %) tienen pocas observaciones: sus métricas tienen
   más varianza.
3. **La importancia de variables no implica causalidad**, y la de impureza (MDI) sesga hacia variables
   continuas; por eso se reporta también la de permutación.
4. **Sin búsqueda de hiperparámetros** (no se usó GridSearch ni validación cruzada) por coste
   computacional: un ajuste fino podría mejorar 1-2 puntos porcentuales.
