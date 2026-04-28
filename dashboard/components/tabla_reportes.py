# File: dashboard/components/tabla_reportes.py
"""Componente Streamlit para mostrar el historial tabular de diagnósticos TC-DIAG."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import streamlit as st

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SEMAFORO_POR_PATOLOGIA = {
    "acv": "🔴",
    "hemorragia_intracraneal": "🔴",
    "desviacion_linea_media": "🟠",
    "fractura_craneal": "🟡",
}


def _resaltar_revision(fila: pd.Series) -> list[str]:
    """Aplica resaltado amarillo a las filas que requieren revisión.

    Args:
        fila: Fila del DataFrame estilizado.

    Returns:
        Lista de estilos CSS por columna.
    """
    if bool(fila.get("requiere_revision", False)):
        return ["background-color: #fff3cd"] * len(fila)
    return [""] * len(fila)


def mostrar_tabla_reportes(df: pd.DataFrame) -> None:
    """Muestra la tabla de diagnósticos con formato clínico y estado de revisión.

    Args:
        df: DataFrame con los diagnósticos recuperados.

    Returns:
        None.
    """
    if df.empty:
        st.info("No hay reportes registrados aún.")
        return

    dataframe = df.copy()
    if "timestamp" in dataframe.columns:
        dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    if "confianza" in dataframe.columns:
        dataframe["confianza"] = pd.to_numeric(dataframe["confianza"], errors="coerce").fillna(0.0)
        dataframe["confianza"] = (dataframe["confianza"] * 100).round(2).astype(str) + "%"

    dataframe["semaforo"] = dataframe["patologia"].map(SEMAFORO_POR_PATOLOGIA).fillna("⚪")
    dataframe["revision"] = dataframe["requiere_revision"].apply(lambda valor: "⚠️ Revisar" if bool(valor) else "✅ OK")

    columnas_mostrar = [
        columna
        for columna in ["timestamp", "report_id", "patologia", "confianza", "semaforo", "revision", "whatsapp_enviado"]
        if columna in dataframe.columns
    ]
    dataframe = dataframe[columnas_mostrar]
    estilo = dataframe.style.apply(_resaltar_revision, axis=1)
    st.dataframe(estilo, use_container_width=True)
