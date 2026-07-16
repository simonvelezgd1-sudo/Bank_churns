"""
Análisis de fuga de clientes (Bank Churners).

Segmentación por límite de crédito y género, pruebas estadísticas
(chi-cuadrado, t-test) y generación de visualizaciones.

Uso:
    python Analisis_TarjetaDeCredito.py
    python Analisis_TarjetaDeCredito.py --data data/BankChurners.csv --out reports/figures
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

# ============================================================
# CONFIGURACIÓN
# ============================================================
BASE_DIR = Path(__file__).resolve().parent

# Nombres de columnas auxiliares generadas por un clasificador Naive Bayes
# que no aportan al análisis exploratorio (se descartan si existen).
COLUMNAS_NAIVE_BAYES = [
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_1",
    "Naive_Bayes_Classifier_Attrition_Flag_Card_Category_Contacts_Count_12_mon_Dependent_count_Education_Level_Months_Inactive_12_mon_2",
]

# Columnas que el resto del script asume que existen; si falta alguna,
# preferimos fallar temprano con un mensaje claro en vez de un KeyError
# a mitad de un cálculo.
COLUMNAS_REQUERIDAS = [
    "Attrition_Flag",
    "Customer_Age",
    "Gender",
    "Credit_Limit",
    "Total_Trans_Ct",
    "Total_Trans_Amt",
    "Months_Inactive_12_mon",
]

CATEGORIAS_FUGA = ["Attrited Customer", "Existing Customer"]


# ============================================================
# ARGUMENTOS DE LÍNEA DE COMANDOS
# ============================================================
def ParsearArgumentos(argv=None):
    parser = argparse.ArgumentParser(description="Análisis de fuga de clientes (Bank Churners)")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Ruta al CSV de entrada (por defecto: data/BankChurners.csv junto al script, "
        "o el primer .csv encontrado junto al script si esa ruta no existe)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Directorio de salida para las gráficas (por defecto: reports/figures junto al script)",
    )
    return parser.parse_args(argv)


def ResolverRutaDatos(ruta_argumento):
    if ruta_argumento:
        ruta = Path(ruta_argumento)
        if not ruta.exists():
            raise FileNotFoundError(f"No se encontró el archivo de datos indicado: {ruta}")
        return ruta

    ruta_por_defecto = BASE_DIR / "data" / "BankChurners.csv"
    if ruta_por_defecto.exists():
        return ruta_por_defecto

    # Compatibilidad con el repo actual, donde el CSV vive junto al script
    # con un nombre distinto (ej. "BankChurners (1).csv").
    candidatos = sorted(BASE_DIR.glob("*.csv"))
    if candidatos:
        return candidatos[0]

    raise FileNotFoundError(
        "No se encontró ningún CSV de entrada. Usá --data para indicar la ruta "
        "o colocá el archivo en data/BankChurners.csv"
    )


def ResolverDirectorioSalida(ruta_argumento):
    directorio = Path(ruta_argumento) if ruta_argumento else (BASE_DIR / "reports" / "figures")
    directorio.mkdir(parents=True, exist_ok=True)
    return directorio


# ============================================================
# CARGA Y LIMPIEZA
# ============================================================
def CargarDatos(ruta):
    df = pd.read_csv(ruta)

    columnas_a_eliminar = [c for c in COLUMNAS_NAIVE_BAYES if c in df.columns]
    if columnas_a_eliminar:
        df = df.drop(columns=columnas_a_eliminar)

    columnas_faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in df.columns]
    if columnas_faltantes:
        raise ValueError(
            f"Faltan columnas requeridas en el dataset: {columnas_faltantes}"
        )

    return df


# ============================================================
# CLASE DE ANÁLISIS (Pandas)
# ============================================================
class AnalizadorClientes:
    def __init__(self, df):
        self.df = df

    def Agrupacion(self):
        Columnas = ['Customer_Age', 'Total_Trans_Ct', 'Credit_Limit']
        resultado = self.df.groupby('Attrition_Flag')[Columnas].agg('mean')
        return resultado

    def SegmentaCredito(self):
        # Comparación fila a fila contra el umbral del cuantil 25,
        # no el escalar del cuantil en sí (ese era el bug original).
        Condicion = self.df['Credit_Limit'] <= self.df['Credit_Limit'].quantile(0.25)
        self.df['Segmento_Credito'] = np.where(Condicion, 'Credito bajo', 'Credito Alto')
        return self.df

    def SegmentoGenero(self):
        Condicion = self.df['Gender'] == 'M'
        self.df['Segmento_Genero'] = np.where(Condicion, 'Masculino', 'Femenino')
        return self.df

    def _TablaCruceConFuga(self, columnas_grupo):
        """
        Tabla de contingencia genérica contra Attrition_Flag, con las dos
        categorías de fuga garantizadas (aunque el grupo no tenga ninguna
        fila de una de ellas) y el porcentaje de fuga calculado.
        """
        Cruce = (
            self.df.groupby(columnas_grupo)['Attrition_Flag']
            .value_counts()
            .unstack(fill_value=0)
            .reindex(columns=CATEGORIAS_FUGA, fill_value=0)
        )
        Cruce['%_Fuga'] = (
            Cruce['Attrited Customer']
            / (Cruce['Attrited Customer'] + Cruce['Existing Customer']).replace(0, np.nan)
        ) * 100
        return Cruce

    def CruzarSegmento(self):
        return self._TablaCruceConFuga(['Segmento_Credito'])

    def CruzarGenero(self):
        return self._TablaCruceConFuga(['Segmento_Genero'])

    def CruzarGenero_Credito(self):
        return self._TablaCruceConFuga(['Segmento_Genero', 'Segmento_Credito'])


# ============================================================
# ANÁLISIS EN CONSOLA (Pandas + NumPy)
# ============================================================
def ImprimirAnalisis(df, analizador, Clientes_Fugados, Clientes_Existentes):
    print("--- Conteo de clientes por estado ---")
    print(df['Attrition_Flag'].value_counts())

    print("\n--- Promedios por grupo (Agrupacion) ---")
    print(analizador.Agrupacion())

    print("\n--- Tabla de contingencia y Porcentaje de Fuga (crédito) ---")
    print(analizador.CruzarSegmento())

    print("\n--- Tabla de contingencia y Porcentaje de Fuga (género) ---")
    print(analizador.CruzarGenero())

    print("\n--- Segmentación por género ---")
    print(analizador.df['Segmento_Genero'].value_counts())

    print("\n--- Cruce género y crédito ---")
    print(analizador.CruzarGenero_Credito())

    # --- NumPy: estadísticas de edad ---
    edades = df["Customer_Age"].to_numpy()
    print("\n--- Estadísticas de edad (NumPy) ---")
    print("Media:", np.mean(edades))
    print("Desv. estándar:", np.std(edades))
    print("Percentil 90:", np.percentile(edades, 90))

    # --- NumPy: array 2D combinado ---
    datos = df[['Customer_Age', 'Credit_Limit', 'Total_Trans_Amt']].to_numpy()
    print("\n--- Array 2D (Customer_Age, Credit_Limit, Total_Trans_Amt) ---")
    print("Shape:", datos.shape)

    # --- NumPy: filtración booleana ---
    credito = df['Credit_Limit'].to_numpy()
    mayores_5000 = credito[credito > 5000]
    print("\n--- Clientes con Credit_Limit > 5000 ---")
    print("Cantidad:", mayores_5000.size)

    # --- NumPy: comparación fugados vs existentes ---
    if len(Clientes_Fugados) == 0 or len(Clientes_Existentes) == 0:
        print("\n--- Comparación de transacciones: Fugados vs Existentes ---")
        print("No hay suficientes datos en alguno de los dos grupos para comparar.")
    else:
        trans_fugados = Clientes_Fugados['Total_Trans_Ct'].to_numpy()
        trans_existentes = Clientes_Existentes['Total_Trans_Ct'].to_numpy()
        diferencia_promedio = trans_fugados.mean() - trans_existentes.mean()
        print("\n--- Comparación de transacciones: Fugados vs Existentes ---")
        print("Diferencia promedio (Fugados - Existentes):", diferencia_promedio)

    print("\n--- Meses inactivo (12m) ---")
    print(df['Months_Inactive_12_mon'].describe())

    # --- Significancia estadística (chi-cuadrado) ---
    print("\n--- Significancia estadística (chi-cuadrado) ---")
    for nombre, Cruce in [
        ("Segmento de crédito", analizador.CruzarSegmento()),
        ("Género", analizador.CruzarGenero()),
    ]:
        tabla = Cruce.drop(columns='%_Fuga')
        if (tabla.sum(axis=1) == 0).any() or tabla.shape[0] < 2:
            print(f"{nombre}: no se pudo calcular (una categoría quedó sin datos)")
            continue
        chi2, p, dof, _ = stats.chi2_contingency(tabla)
        print(f"{nombre}: chi2={chi2:.2f}, p-value={p:.4f}")

    # --- Significancia estadística (t-test) para variables continuas ---
    print("\n--- Significancia estadística: t-test (continuas) ---")
    if len(Clientes_Fugados) == 0 or len(Clientes_Existentes) == 0:
        print("No hay suficientes datos en alguno de los dos grupos para el t-test.")
    else:
        variables_continuas = ['Customer_Age', 'Total_Trans_Ct', 'Credit_Limit']
        for variable in variables_continuas:
            grupo_fugado = Clientes_Fugados[variable].to_numpy()
            grupo_existente = Clientes_Existentes[variable].to_numpy()
            t_stat, p_valor = stats.ttest_ind(grupo_fugado, grupo_existente, equal_var=False)
            significativo = "sí" if p_valor < 0.05 else "no"
            print(f"{variable}: t={t_stat:.2f}, p-value={p_valor:.4f} (significativo: {significativo})")


# ============================================================
# VISUALIZACIÓN (7 gráficas, un solo flujo)
# ============================================================
def GenerarGraficas(df, analizador, output_dir):
    # 1) Distribución de edades
    plt.figure(figsize=(11, 6))
    sns.histplot(data=df, x='Customer_Age', bins=30, kde=True, color='blue')
    plt.title('Distribución de Edades de Clientes', fontsize=16)
    plt.xlabel('Edad', fontsize=14)
    plt.ylabel('Frecuencia', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig(output_dir / 'Distribucion_Edades.png', dpi=300)
    plt.close()

    # 2) Transacciones: fugados vs existentes
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Attrition_Flag', y='Total_Trans_Ct', data=df, palette='Set2', hue='Attrition_Flag')
    plt.title('Comparación de Transacciones entre Clientes Fugados y Existentes', fontsize=16)
    plt.xlabel('Estado del Cliente', fontsize=14)
    plt.ylabel('Total de Transacciones', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / 'Comparacion_Transacciones.png', dpi=300)
    plt.close()

    # 3) % de fuga por segmento de crédito
    plt.figure(figsize=(10, 6))
    cruce_segmento = analizador.CruzarSegmento()
    cruce_segmento['%_Fuga'].plot(kind='bar', color=['cyan', 'black'])
    plt.xlabel('Segmento de Crédito', fontsize=14)
    plt.ylabel('Porcentaje de Fuga', fontsize=14)
    plt.title('Comparación de Fuga por Segmento de Crédito', fontsize=16)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / 'Comparacion_Segmentos.png', dpi=300)
    plt.close()

    # 4) % de fuga por género
    plt.figure(figsize=(10, 6))
    cruce_genero = analizador.CruzarGenero()
    cruce_genero['%_Fuga'].plot(kind='bar', color=['lightblue', 'lightpink'])
    plt.xlabel('Género', fontsize=14)
    plt.ylabel('Porcentaje de Fuga', fontsize=14)
    plt.title('Comparación de Fuga por Género', fontsize=16)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'Comparacion_Genero.png', dpi=300)
    plt.close()

    # 5) Promedios por estado del cliente
    resultado = analizador.Agrupacion().reset_index()
    plot1_data = resultado.melt(
        id_vars="Attrition_Flag",
        value_vars=["Customer_Age", "Total_Trans_Ct", "Credit_Limit"],
        var_name="Variable",
        value_name="Promedio",
    )
    plt.figure(figsize=(12, 6))
    sns.barplot(
        data=plot1_data,
        x="Variable",
        y="Promedio",
        hue="Attrition_Flag",
        palette=["#4C69A8", "#F58518"],
    )
    plt.title("Promedio de variables clave por estado del cliente", fontsize=16)
    plt.xlabel("Variable", fontsize=12)
    plt.ylabel("Promedio", fontsize=12)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_dir / "Promedios_por_estado.png", dpi=300)
    plt.close()

    # 6) Heatmap % de fuga por género y segmento de crédito
    cruce = analizador.CruzarGenero_Credito().reset_index()
    pivot = cruce.pivot(index="Segmento_Genero", columns="Segmento_Credito", values="%_Fuga")
    plt.figure(figsize=(10, 6))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        linewidths=0.5,
        cbar_kws={"label": "% de fuga"},
    )
    plt.title("Porcentaje de fuga por género y segmento de crédito", fontsize=16)
    plt.xlabel("Segmento de crédito", fontsize=12)
    plt.ylabel("Género", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "Fuga_por_genero_y_credito.png", dpi=300)
    plt.close()

    # 7) Gráfico del test de chi-cuadrado para el segmento de crédito
    plt.figure(figsize=(8, 5))
    cruce_segmento = analizador.CruzarSegmento()
    cruce_segmento.drop(columns='%_Fuga').plot(kind='bar', stacked=True, color=["#00E1FF", "#F51818"])
    plt.title('Chi-square Test for Credit Segment')
    plt.xlabel('Credit Segment')
    plt.ylabel('Number of Customers')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_dir / 'Chi_square_Credit_Segment.png', dpi=300)
    plt.close()

    nombres_generados = [
        "Distribucion_Edades.png",
        "Comparacion_Transacciones.png",
        "Comparacion_Segmentos.png",
        "Comparacion_Genero.png",
        "Promedios_por_estado.png",
        "Fuga_por_genero_y_credito.png",
        "Chi_square_Credit_Segment.png",
    ]
    print("\n--- Gráficos generados correctamente ---")
    for nombre in nombres_generados:
        print(f"- {output_dir / nombre}")


# ============================================================
# FLUJO PRINCIPAL
# ============================================================
def main(argv=None):
    argumentos = ParsearArgumentos(argv)

    try:
        ruta_datos = ResolverRutaDatos(argumentos.data)
    except FileNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    directorio_salida = ResolverDirectorioSalida(argumentos.out)

    try:
        df = CargarDatos(ruta_datos)
    except ValueError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    analizador = AnalizadorClientes(df)
    analizador.SegmentaCredito()
    analizador.SegmentoGenero()

    Clientes_Fugados = df[df["Attrition_Flag"] == 'Attrited Customer']
    Clientes_Existentes = df[df["Attrition_Flag"] == 'Existing Customer']

    ImprimirAnalisis(df, analizador, Clientes_Fugados, Clientes_Existentes)
    GenerarGraficas(df, analizador, directorio_salida)
    return 0


if __name__ == "__main__":
    sys.exit(main())


