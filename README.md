# Bank_churns — Análisis crítico y plan de organización

Proyecto: Análisis de fuga de clientes (Bank Churners)

Este repositorio contiene un análisis exploratorio y visual del dataset BankChurners. A continuación se presenta una propuesta de organización del repositorio, un resumen crítico del código principal (Analisis_TarjetaDeCredito.py), instrucciones para ejecutar el proyecto, y recomendaciones/próximos pasos.

## Qué es (resumen corto)
Análisis exploratorio centrado en segmentación por límite de crédito y género, con pruebas estadísticas (chi-cuadrado, t-test) y generación de visualizaciones que apoyan decisiones para reducir la fuga de clientes.

### Stack
- Language(s): Python (principal), HTML (dashboard estático)
- Framework / runtime: Python 3.x
- Notable libraries: pandas, numpy, matplotlib, seaborn, scipy

## Propuesta de organización (árbol de directorios)
```
README.md                       # este archivo (actualizado)
REPO_STRUCTURE.md               # comandos sugeridos para reordenar (creado)
requirements.txt                # dependencias (recomendado)

data/                           # datos crudos (CSV)
  BankChurners.csv

src/                            # código fuente del proyecto
  analysis/
    Analisis_TarjetaDeCredito.py

notebooks/                       # Jupyter notebooks (exploración iterativa)

reports/
  figures/                       # imágenes y gráficos generados
    Distribucion_Edades.png
    Comparacion_Transacciones.png
    Comparacion_Segmentos.png
    Comparacion_Genero.png
    Promedios_por_estado.png
    Fuga_por_genero_y_credito.png
  dashboard/                      # dashboard HTML o assets
    dasboard_interactivo.html

docs/                            # documentación ampliada / metodología

scripts/                         # utilidades / scripts para preparar datos

.gitignore
```

## Comandos sugeridos para reorganizar (ejecutar desde la raíz del repo)
(Copiar/pegar en tu máquina; esto preserva historial si usas `git mv`)

```
mkdir -p data src/analysis reports/figures notebooks docs scripts
git mv "Analisis_TarjetaDeCredito.py" src/analysis/Analisis_TarjetaDeCredito.py
git mv "BankChurners (1).csv" data/BankChurners.csv
git mv dasboard\ _interactivo.html reports/dashboard/dasboard_interactivo.html || mv "dasboard _interactivo.html" reports/dashboard/dasboard_interactivo.html
git mv *.png reports/figures/ || true
git add . && git commit -m "Reorganizar: mover código, datos y figuras a carpetas por convención"
```

Notas:
- Si prefieres no mover archivos inmediatamente, crea las carpetas y mantén los nombres actuales; el README y REPO_STRUCTURE.md serán guía de referencia.
- Asegúrate de actualizar las rutas absolutas en el script (DATA_PATH y OUTPUT_DIR) o convertir el script para aceptar argumentos.

## Análisis crítico del código (Analisis_TarjetaDeCredito.py)
Esto no es un resumen amable: son observaciones concretas, su impacto y acciones recomendadas.

Observaciones críticas y correcciones propuestas

- Rutas absolutas y reproducibilidad
  - Problema: DATA_PATH y OUTPUT_DIR están hardcodeados a rutas de una máquina local (/home/velez/Practicas_Numpy-Pandas/...). Esto impide ejecutar el script en otra máquina.
  - Impacto: baja reproducibilidad, riesgo de fallos al correr el script y al generar/sobrescribir outputs.
  - Recomendación: usar argparse para aceptar --data y --out, o leer variables de entorno; usar rutas relativas dentro de `data/` y `reports/`.

- Uso incorrecto de np.where en SegmentaCredito
  - Problema: SegmentaCredito calcula `condicion = self.df['Credit_Limit'].quantile(0.25)` (un escalar) y luego `np.where(condicion, 'Credito bajo', 'Credito Alto')`. np.where espera una condición booleana por fila; usando un escalar siempre evalúa True salvo que sea 0/False.
  - Impacto: la columna `Segmento_Credito` queda mal calculada (probablemente siempre 'Credito bajo' o siempre 'Credito Alto').
  - Recomendación: cambiar a `self.df['Credit_Limit'] <= condicion` como la condición booleana, o usar pd.cut para crear segmentos cuantílicos.

- Dependencia de nombres de columna exactos y eliminación de columnas
  - Problema: la lista COLUMNAS_NAIVE_BAYES se elimina sin comprobar si existen; si las columnas no están presentes el script fallará.
  - Recomendación: envolver en try/except o usar df.drop(..., errors='ignore'). Validar columnas esperadas al inicio.

- Asunciones en tablas pivote y nombres de columnas
  - Problema: al hacer unstack() y luego indexar por 'Attrited Customer' y 'Existing Customer' se asume que esos nombres existen y en ese orden.
  - Impacto: si los labels cambian o el dataset se limpia, aparecen KeyError o resultados incorrectos.
  - Recomendación: usar .reindex(columns=[..], fill_value=0) o comprobar columnas antes de operar; nombrar explícitamente variables.

- Mensajes y salida a consola
  - Observación: buen nivel de salida; sin embargo mezclar análisis y generación de figuras en un mismo flujo reduce flexibilidad para tests o uso programático.
  - Recomendación: separar funciones puras (cálculos) de efectos (imprimir/guardar figuras). Añadir pruebas unitarias para funciones clave (SegmentaCredito, CruzarGenero, etc.).

- Buenas prácticas y calidad del código
  - Sugerencia: convertir la clase AnalizadorClientes en una clase con métodos que no modifican el df in-place (retornar copies cuando sea necesario) o documentar claramente los efectos.
  - Añadir un requirements.txt con versiones (por ejemplo: pandas>=1.5, numpy>=1.25, seaborn>=0.12, scipy>=1.10).
  - Añadir un archivo .py con funciones auxiliares y tests en tests/.

## Logros e insights (lo que realmente aporta este proyecto)
Presento los logros y hallazgos importantes que se pueden extraer con el código actual y cómo maximizar su valor:

- Logros técnicos
  - Pipeline básico end-to-end: desde carga de CSV hasta generación de 6 visualizaciones y tests estadísticos automatizados (chi-cuadrado y t-test). Esto demuestra comprensión del flujo EDA -> hipótesis -> visualización.
  - Uso combinado de pandas y NumPy para análisis estadístico y vectorizado, lo cual facilita performance en datasets medianos.
  - Producción de outputs reproducibles (gráficos guardados en disco) que son directamente utilizables para reportes.

- Insights accionables (ejemplos que tu análisis permite descubrir)
  - Segmentación por crédito y género: con la matriz cruce y el heatmap se puede identificar segmentos con mayor probabilidad de fuga (objetivo para campañas de retención).
  - Diferencias en conteo de transacciones: el t-test comparativo entre fugados y existentes delimita si el comportamiento transaccional es un predictor válido.
  - Meses inactivos y límites de crédito: variables con potencial para modelos predictivos — el proyecto está en posición de pasar a modelado supervisado para predicción de churn.

## Cómo ejecutar (ruta corta)
Requisitos: Python 3.9+ (recomiendo 3.10+), instalar dependencias en entorno virtual.

```bash
python -m venv .venv
source .venv/bin/activate   # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
```

Si sigues la reorganización sugerida (mover archivos):
```bash
python src/analysis/Analisis_TarjetaDeCredito.py
```

Si no mueves archivos, y antes de ejecutar, edita el script para usar rutas relativas:
- Cambia DATA_PATH a `Path('data/BankChurners.csv')`
- Cambia OUTPUT_DIR a `Path('reports/figures')`

## Próximos pasos recomendados (priorizados)
1. Parametrizar el script (argparse) para hacer reproducible la ejecución en CI/CD y local.
2. Añadir requirements.txt y un Makefile con comandos: `make install`, `make run`, `make figures`.
3. Escribir tests unitarios para las funciones de segmentación y tablas de cruce.
4. Considerar un notebook en `notebooks/` describiendo hallazgos y pasos de limpieza reproducibles.

---

Si quieres, puedo:
- Aplicar la reorganización automáticamente (mover archivos usando git mv y subir los cambios). 
- Refactorizar Analisis_TarjetaDeCredito.py para aceptar argumentos, corregir errores identificados y añadir un small CLI.
- Generar requirements.txt y un Makefile.

Indica cuál de estas acciones prefieres y lo haré ahora.
