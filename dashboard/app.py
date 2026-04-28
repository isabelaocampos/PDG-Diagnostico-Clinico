# File: dashboard/app.py
"""Aplicación principal Streamlit para visualizar el ecosistema TC-DIAG."""

from __future__ import annotations

import logging
import os
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

from components.distribucion_patologias import mostrar_distribucion, mostrar_tendencia_temporal
from components.metricas import mostrar_metricas
from components.tabla_reportes import mostrar_tabla_reportes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()
DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/tc_diag")

st.set_page_config(
    title="TC-DIAG Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

try:
    from streamlit_autorefresh import st_autorefresh
except Exception:  # noqa: BLE001
    st_autorefresh = None


@st.cache_data(ttl=30)
def cargar_datos(date_from: date, date_to: date) -> pd.DataFrame:
    """Carga diagnósticos desde PostgreSQL dentro de un rango temporal.

    Args:
        date_from: Fecha de inicio del filtro.
        date_to: Fecha de fin del filtro.

    Returns:
        DataFrame con los registros recuperados.
    """
    engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    inicio = pd.Timestamp(date_from).to_pydatetime().replace(tzinfo=None)
    fin_exclusivo = (pd.Timestamp(date_to) + pd.Timedelta(days=1)).to_pydatetime().replace(tzinfo=None)
    consulta = text(
        """
        SELECT *
        FROM diagnosticos
        WHERE timestamp >= :inicio
          AND timestamp < :fin_exclusivo
        ORDER BY timestamp DESC
        """
    )
    try:
        with engine.connect() as conexion:
            dataframe = pd.read_sql(consulta, conexion, params={"inicio": inicio, "fin_exclusivo": fin_exclusivo})
        return dataframe
    except Exception as exc:  # noqa: BLE001
        logger.error("No fue posible cargar datos del dashboard: %s", exc)
        return pd.DataFrame()


if st_autorefresh is not None:
    st_autorefresh(interval=30000, key="tcdiag_autorefresh")

st.title("TC-DIAG Dashboard")
st.caption("Monitoreo clínico de diagnósticos radiológicos multicategoría")

with st.sidebar:
    st.title("TC-DIAG")
    st.write("Panel de seguimiento de diagnósticos radiológicos críticos.")

    fecha_hoy = date.today()
    fecha_inicio_defecto = fecha_hoy - timedelta(days=7)
    rango_fechas = st.date_input(
        "Rango de fechas",
        value=(fecha_inicio_defecto, fecha_hoy),
        max_value=fecha_hoy,
    )
    if isinstance(rango_fechas, tuple) and len(rango_fechas) == 2:
        date_from, date_to = rango_fechas
    else:
        date_from = fecha_inicio_defecto
        date_to = fecha_hoy

    patologias_disponibles = ["acv", "hemorragia_intracraneal", "desviacion_linea_media", "fractura_craneal"]
    patologias_seleccionadas = st.multiselect(
        "Patologías",
        options=patologias_disponibles,
        default=patologias_disponibles,
    )
    confianza_minima = st.slider("Umbral mínimo de confianza", 0.0, 1.0, 0.0, 0.01)

    if st.button("Actualizar ahora"):
        st.cache_data.clear()
        st.rerun()


df = cargar_datos(date_from, date_to)
if not df.empty:
    if patologias_seleccionadas:
        df = df[df["patologia"].isin(patologias_seleccionadas)]
    df = df[df["confianza"].fillna(0.0) >= confianza_minima]

mostrar_metricas(df)

st.divider()
col_izquierda, col_derecha = st.columns([3, 2])
with col_izquierda:
    mostrar_distribucion(df)
with col_derecha:
    mostrar_tendencia_temporal(df)

st.divider()
st.subheader("Últimos Diagnósticos")
mostrar_tabla_reportes(df)
