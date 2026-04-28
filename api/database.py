"""Persistencia de diagnósticos TC-DIAG con SQLAlchemy.

Compatible con SQLite (desarrollo local) y PostgreSQL (producción).
La URL de conexión se define exclusivamente en la variable DATABASE_URL del .env.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Generator, List, Optional

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

load_dotenv()

logger = logging.getLogger(__name__)

DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./tc_diag.db")

# SQLite requiere check_same_thread=False para funcionar con FastAPI.
_args_conexion = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
motor = create_engine(DATABASE_URL, pool_pre_ping=True, connect_args=_args_conexion)
SesionLocal = sessionmaker(autocommit=False, autoflush=False, bind=motor)
Base = declarative_base()


class Diagnostico(Base):
    """Registro de un informe radiológico procesado por TC-DIAG."""

    __tablename__ = "diagnosticos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(100), nullable=False, index=True)
    es_critico = Column(Boolean, nullable=False, default=False)
    triage_score = Column(Float, nullable=True)
    patologia = Column(String(50), nullable=True)
    icd10 = Column(String(10), nullable=True)
    confianza = Column(Float, nullable=True)
    prob_acv = Column(Float, nullable=True)
    prob_hemorragia_intracraneal = Column(Float, nullable=True)
    prob_desviacion_linea_media = Column(Float, nullable=True)
    prob_fractura_craneal = Column(Float, nullable=True)
    requiere_revision = Column(Boolean, default=False)
    whatsapp_enviado = Column(Boolean, default=False)
    marca_temporal = Column(DateTime, default=lambda: datetime.now(timezone.utc))


def obtener_sesion() -> Generator[Session, None, None]:
    """Genera una sesión de base de datos para inyección en FastAPI."""
    sesion = SesionLocal()
    try:
        yield sesion
    finally:
        sesion.close()


def guardar_diagnostico(
    sesion: Session,
    report_id: str,
    es_critico: bool,
    triage_score: float,
    patologia: Optional[str],
    icd10: Optional[str],
    probabilidades: dict,
    confianza: float,
    requiere_revision: bool,
    whatsapp_enviado: bool = False,
) -> Diagnostico:
    """Inserta un diagnóstico y retorna el registro persistido."""
    registro = Diagnostico(
        report_id=report_id,
        es_critico=es_critico,
        triage_score=triage_score,
        patologia=patologia,
        icd10=icd10,
        confianza=confianza,
        prob_acv=probabilidades.get("acv"),
        prob_hemorragia_intracraneal=probabilidades.get("hemorragia_intracraneal"),
        prob_desviacion_linea_media=probabilidades.get("desviacion_linea_media"),
        prob_fractura_craneal=probabilidades.get("fractura_craneal"),
        requiere_revision=requiere_revision,
        whatsapp_enviado=whatsapp_enviado,
    )
    try:
        sesion.add(registro)
        sesion.commit()
        sesion.refresh(registro)
        return registro
    except IntegrityError as exc:
        sesion.rollback()
        logger.error("Error al guardar el diagnóstico '%s': %s", report_id, exc)
        raise


def obtener_historial(
    sesion: Session,
    limite: int = 50,
    patologia: Optional[str] = None,
    solo_criticos: Optional[bool] = None,
) -> List[Diagnostico]:
    """Recupera diagnósticos ordenados por fecha descendente."""
    consulta = sesion.query(Diagnostico)
    if patologia is not None:
        consulta = consulta.filter(Diagnostico.patologia == patologia)
    if solo_criticos is not None:
        consulta = consulta.filter(Diagnostico.es_critico == solo_criticos)
    return consulta.order_by(desc(Diagnostico.marca_temporal)).limit(limite).all()


try:
    Base.metadata.create_all(bind=motor)
except Exception as exc:
    logger.error("No se pudieron crear las tablas: %s", exc)
