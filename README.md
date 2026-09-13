# Actividad 2 — Modelos Supervisados Avanzados y Aprendizaje No Supervisado

**Asignatura:** Aprendizaje Automático — Especialización en Inteligencia Artificial (UNIR)
**Autor:** Omar Javier Daza Leguizamón
**Notebook:** `Act2_Housing_UNIR_V1.ipynb`
**Fecha:** 13/09/2026

## Conjunto de datos

Viviendas residenciales vendidas en Ames, Iowa, entre 2006 y 2010 (De Cock, 2011),
publicado en Kaggle como *USA Housing Dataset*.

- `housing_train.csv` — 1.460 registros y 81 columnas. **Es el único archivo usado para
  entrenar y evaluar**, dividido internamente en 80 % de entrenamiento y 20 % de prueba.
- `housing_test.csv` — 1.459 registros y 80 columnas. **No contiene la columna
  `SalePrice`**, por lo que no permite calcular RMSE ni matrices de confusión. En el
  notebook se lee únicamente en la sección 0.4, para documentar esa limitación.

La partición interna produce un conjunto de prueba de 292 registros con 25, 265 y 2
viviendas en los grupos 1, 2 y 3, cifras que coinciden con las matrices de confusión
del enunciado. Esto confirma que las cifras de referencia del curso se obtuvieron
sobre `housing_train.csv`.

## Estructura del proyecto

```
Act2_Housing/
├── Act2_Housing_UNIR_V1.ipynb   # notebook principal
├── requirements.txt
├── README.md
├── verificar_entorno.py         # comprobación del entorno antes de ejecutar
├── data/
│   ├── housing_train.csv
│   └── housing_test.csv
├── figuras/                     # 16 imágenes PNG generadas al ejecutar
└── salidas/                     # 13 tablas CSV y TXT generadas al ejecutar
```

Las carpetas `figuras/` y `salidas/` las crea el notebook automáticamente. La carpeta
`data/` debe existir con los dos archivos antes de ejecutar.

## Instalación del entorno

**Requiere Python 3.12.** TensorFlow publica ruedas para Python 3.10 a 3.13 únicamente;
con Python 3.14 la instalación falla con `No matching distribution found for tensorflow`.

El entorno virtual debe crearse **fuera de una carpeta sincronizada con OneDrive**. Las
librerías con extensiones nativas, como XGBoost y TensorFlow, cargan archivos DLL que la
política WDAC de Windows bloquea cuando residen en una ruta sincronizada.

```powershell
# Instalar Python 3.12 si no está presente (convive con otras versiones)
winget install -e --id Python.Python.3.12

# Crear el entorno en una ruta local, no sincronizada
mkdir C:\dev\act2 ; cd C:\dev\act2
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python --version                 # debe responder 3.12.x
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name act2 --display-name "Python (Act2)"
```

Verificación previa:

```powershell
python verificar_entorno.py
```

El script comprueba la versión de Python, la ubicación del entorno virtual y la carga
efectiva de XGBoost y TensorFlow. Si alguna de las dos falla, indica la sustitución de
respaldo correspondiente (`HistGradientBoosting` y `MLPRegressor` de scikit-learn).

## Ejecución

Abrir `Act2_Housing_UNIR_V1.ipynb` con el kernel «Python (Act2)» y ejecutar
**Restart & Run All**. El notebook corre de principio a fin sin intervención y es
reproducible con `RANDOM_STATE = 42`. El tiempo aproximado de ejecución es de cinco a
diez minutos; la sección más lenta es el barrido del parámetro C en la 4.6.

Al ejecutar la red neuronal aparece un aviso indicando que TensorFlow usará CPU en lugar
de GPU. Es informativo y no afecta el resultado.

## Nota sobre la lectura de los datos

Los archivos se leen con:

```python
pd.read_csv(ruta, keep_default_na=False, na_values=["NA", ""])
```

La configuración predeterminada de pandas interpreta la cadena `None` como valor nulo, y
en este conjunto `None` es una categoría legítima de `MasVnrType` que agrupa 864
viviendas sin revestimiento de mampostería. Sin este parámetro, el conteo de faltantes de
esa columna pasa de 8 a 872 y el diagnóstico de la sección 2 queda distorsionado. Con la
lectura indicada, las cifras coinciden con las de la Figura 4 del enunciado.

## Correspondencia con los criterios de evaluación

| Sección del notebook | Criterio | Peso |
|---|---|---|
| 0. Preparación previa a los análisis de datos | — | — |
| 1. Análisis descriptivo de los datos | 1 | 10 % |
| 2. Tratamiento de valores faltantes | 2 | 10 % |
| 3. Problema de regresión | 3 | 15 % |
| 4. Problema de clasificación | 4 | 15 % |
| 5. Agrupamiento para segmentación | 5 | 15 % |
| 6. Detección de anomalías | 6 | 10 % |
| 7. Aprendizaje por refuerzo | 7 | 10 % |
| 8. Comentarios sobre los resultados | 8 | 15 % |

La numeración del informe es la misma del notebook, de modo que cada resultado del
informe puede verificarse en la sección homóloga.

## Resultados principales

**Regresión** (RMSE sobre el conjunto de prueba, precio promedio 181.978 USD):
XGBoost 24.304,6 · red neuronal 27.236,3 · random forest 29.615,4 · árbol podado
37.226,8 · árbol sin podar 38.816,5.

**Clasificación** (exactitud balanceada): SVM lineal 0,768 · árbol con ponderación de
clases 0,698 · SVM RBF 0,619 · árbol 0,580 · XGBoost 0,579 · random forest 0,476.

**No supervisado:** K-means con K = 3 y agrupamiento jerárquico de Ward; Isolation
Forest señala 30 viviendas atípicas, entre ellas el 77,8 % de las del grupo 3.

**Refuerzo:** el agente Q-learning resuelve el entorno FrozenLake en 100 de 100
episodios de evaluación.

## Referencia del conjunto de datos

De Cock, D. (2011). Ames, Iowa: Alternative to the Boston housing data as an end of
semester regression project. *Journal of Statistics Education*, 19(3).
Documentación de variables: `https://jse.amstat.org/v19n3/decock/DataDocumentation.txt`
