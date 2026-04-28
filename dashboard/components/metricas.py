# File: dashboard/components/metricas.py
"""Componente Streamlit para mostrar indicadores clave del sistema TC-DIAG."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mostrar_metricas(df: pd.DataFrame) -> None:
    """Muestra tarjetas KPI con la distribución de diagnósticos.

    Args:
        df: DataFrame filtrado con diagnósticos.

    Returns:
        None.
    """
    total = int(len(df))
    conteos = df["patologia"].value_counts() if not df.empty and "patologia" in df.columns else pd.Series(dtype=int)
    patologias = ["acv", "hemorragia_intracraneal", "desviacion_linea_media", "fractura_craneal"]
    etiquetas = ["ACV", "Hemorragia IC", "Desviación LM", "Fractura"]

    columnas = st.columns(5)
    columnas[0].metric("Total Reportes", total)

    for indice, (patologia, etiqueta) in enumerate(zip(patologias, etiquetas), start=1):
        cantidad = int(conteos.get(patologia, 0))
        porcentaje = (cantidad / total * 100) if total > 0 else 0.0
        columnas[indice].metric(etiqueta, cantidad, f"{porcentaje:.1f}%")

    requeridos = int(df["requiere_revision"].sum()) if not df.empty and "requiere_revision" in df.columns else 0
    st.warning(f"⚠️ Requieren Revisión: {requeridos}")
