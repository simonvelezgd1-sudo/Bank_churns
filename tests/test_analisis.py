"""
Tests unitarios para Analisis_TarjetaDeCredito.py

Cubre especialmente las funciones que el README señalaba como riesgosas:
SegmentaCredito (el bug histórico de np.where con un escalar),
las tablas de cruce (CruzarSegmento, CruzarGenero, CruzarGenero_Credito)
y la carga/validación de columnas en CargarDatos.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from Analisis_TarjetaDeCredito import (
    AnalizadorClientes,
    CargarDatos,
    COLUMNAS_NAIVE_BAYES,
    ResolverRutaDatos,
)


@pytest.fixture
def df_ejemplo():
    """
    12 clientes con Credit_Limit variado (para que el cuantil 25 caiga
    en un punto conocido) y ambas categorías de Attrition_Flag y Gender.
    """
    return pd.DataFrame(
        {
            "Attrition_Flag": [
                "Existing Customer", "Existing Customer", "Existing Customer",
                "Existing Customer", "Existing Customer", "Existing Customer",
                "Existing Customer", "Existing Customer", "Existing Customer",
                "Attrited Customer", "Attrited Customer", "Attrited Customer",
            ],
            "Customer_Age": [30, 35, 40, 45, 50, 55, 60, 65, 70, 33, 44, 55],
            "Gender": ["M", "F", "M", "F", "M", "F", "M", "F", "M", "F", "M", "F"],
            "Credit_Limit": [1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 1200, 2200, 3200],
            "Total_Trans_Ct": [10, 20, 30, 40, 50, 60, 70, 80, 90, 15, 25, 35],
            "Total_Trans_Amt": [100, 200, 300, 400, 500, 600, 700, 800, 900, 150, 250, 350],
            "Months_Inactive_12_mon": [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3],
        }
    )


class TestSegmentaCredito:
    def test_produce_dos_segmentos_distintos(self, df_ejemplo):
        analizador = AnalizadorClientes(df_ejemplo.copy())
        resultado = analizador.SegmentaCredito()
        segmentos_unicos = resultado["Segmento_Credito"].unique()
        assert len(segmentos_unicos) == 2
        assert set(segmentos_unicos) == {"Credito bajo", "Credito Alto"}

    def test_respeta_el_umbral_del_cuantil_25(self, df_ejemplo):
        analizador = AnalizadorClientes(df_ejemplo.copy())
        resultado = analizador.SegmentaCredito()
        umbral = df_ejemplo["Credit_Limit"].quantile(0.25)

        bajo = resultado[resultado["Segmento_Credito"] == "Credito bajo"]
        alto = resultado[resultado["Segmento_Credito"] == "Credito Alto"]

        assert (bajo["Credit_Limit"] <= umbral).all()
        assert (alto["Credit_Limit"] > umbral).all()

    def test_no_devuelve_un_solo_valor_para_todos(self, df_ejemplo):
        analizador = AnalizadorClientes(df_ejemplo.copy())
        resultado = analizador.SegmentaCredito()
        conteo = resultado["Segmento_Credito"].value_counts()
        assert conteo.get("Credito Alto", 0) > 0
        assert conteo.get("Credito bajo", 0) > 0


class TestSegmentoGenero:
    def test_asigna_masculino_femenino_correctamente(self, df_ejemplo):
        analizador = AnalizadorClientes(df_ejemplo.copy())
        resultado = analizador.SegmentoGenero()
        esperado = df_ejemplo["Gender"].map({"M": "Masculino", "F": "Femenino"})
        assert (resultado["Segmento_Genero"] == esperado).all()


class TestTablasDeCruce:
    def test_cruzar_segmento_tiene_ambas_categorias_de_fuga(self, df_ejemplo):
        analizador = AnalizadorClientes(df_ejemplo.copy())
        analizador.SegmentaCredito()
        cruce = analizador.CruzarSegmento()
        assert "Attrited Customer" in cruce.columns
        assert "Existing Customer" in cruce.columns
        assert "%_Fuga" in cruce.columns

    def test_cruzar_segmento_no_falla_si_un_grupo_no_tiene_fugados(self):
        df = pd.DataFrame(
            {
                "Attrition_Flag": ["Existing Customer"] * 5,
                "Credit_Limit": [1000, 2000, 3000, 4000, 5000],
                "Gender": ["M", "F", "M", "F", "M"],
            }
        )
        analizador = AnalizadorClientes(df)
        analizador.SegmentaCredito()
        cruce = analizador.CruzarSegmento()
        assert (cruce["Attrited Customer"] == 0).all()

    def test_porcentaje_de_fuga_calculado_correctamente(self, df_ejemplo):
        analizador = AnalizadorClientes(df_ejemplo.copy())
        analizador.SegmentoGenero()
        cruce = analizador.CruzarGenero()
        for _, fila in cruce.iterrows():
            total = fila["Attrited Customer"] + fila["Existing Customer"]
            esperado = (fila["Attrited Customer"] / total) * 100
            assert fila["%_Fuga"] == pytest.approx(esperado)

    def test_cruzar_genero_credito_combina_ambos_segmentos(self, df_ejemplo):
        analizador = AnalizadorClientes(df_ejemplo.copy())
        analizador.SegmentaCredito()
        analizador.SegmentoGenero()
        cruce = analizador.CruzarGenero_Credito()
        assert cruce.index.nlevels == 2


class TestCargarDatos:
    def test_elimina_columnas_naive_bayes_si_existen(self, tmp_path, df_ejemplo):
        df_con_naive = df_ejemplo.copy()
        for columna in COLUMNAS_NAIVE_BAYES:
            df_con_naive[columna] = 0.0
        ruta_csv = tmp_path / "datos.csv"
        df_con_naive.to_csv(ruta_csv, index=False)

        resultado = CargarDatos(ruta_csv)
        for columna in COLUMNAS_NAIVE_BAYES:
            assert columna not in resultado.columns

    def test_no_falla_si_columnas_naive_bayes_no_existen(self, tmp_path, df_ejemplo):
        ruta_csv = tmp_path / "datos.csv"
        df_ejemplo.to_csv(ruta_csv, index=False)
        resultado = CargarDatos(ruta_csv)
        assert len(resultado) == len(df_ejemplo)

    def test_lanza_error_claro_si_falta_columna_requerida(self, tmp_path, df_ejemplo):
        df_incompleto = df_ejemplo.drop(columns=["Credit_Limit"])
        ruta_csv = tmp_path / "datos.csv"
        df_incompleto.to_csv(ruta_csv, index=False)

        with pytest.raises(ValueError, match="Credit_Limit"):
            CargarDatos(ruta_csv)


class TestResolverRutaDatos:
    def test_lanza_error_si_ruta_indicada_no_existe(self):
        with pytest.raises(FileNotFoundError):
            ResolverRutaDatos("/ruta/que/no/existe.csv")
