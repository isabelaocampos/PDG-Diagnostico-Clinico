# File: api/database.py
"""Persistencia PostgreSQL para diagnósticos del sistema TC-DIAG."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Generator, List, Optional

from dotenv import load_dotenv
from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, create_engine, desc
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

DATABASE_URL: str = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/tc_diag")
engine = create_engine(DATABASE_URL, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Diagnostico(Base):
    """Entidad ORM para almacenar el diagnóstico multiclase de un reporte."""

    __tablename__ = "diagnosticos"

    id = Column(Integer, primary_key=True, autoincrement=True)
    report_id = Column(String(100), nullable=False, index=True)
    patologia = Column(String(50), nullable=False)
    confianza = Column(Float, nullable=False)
    prob_acv = Column(Float)
    prob_hemorragia_intracraneal = Column(Float)
    prob_desviacion_linea_media = Column(Float)
    prob_fractura_craneal = Column(Float)
    requiere_revision = Column(Boolean, default=False)
    whatsapp_enviado = Column(Boolean, default=False)
    timestamp = Column(DateTime, default=datetime.utcnow)


def get_db() -> Generator[Session, None, None]:
    """Genera una sesión de base de datos para FastAPI.

    Args:
        No aplica.

    Returns:
        Generador de sesiones SQLAlchemy.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def guardar_diagnostico(
    db: Session,
    report_id: str,
    patologia: str,
    probabilidades: dict,
    confianza: float,
    requiere_revision: bool,
    whatsapp_enviado: bool = False,
) -> Diagnostico:
    """Inserta un diagnóstico en la base de datos.

    Args:
        db: Sesión SQLAlchemy activa.
        report_id: Identificador único del reporte.
        patologia: Patología predicha.
        probabilidades: Mapa de probabilidades por clase.
        confianza: Confianza de la predicción.
        requiere_revision: Bandera de revisión clínica.
        whatsapp_enviado: Indica si la notificación fue enviada.

    Returns:
        Instancia persistida de Diagnostico.
    """
    diagnostico = Diagnostico(
        report_id=report_id,
        patologia=patologia,
        confianza=confianza,
        prob_acv=probabilidades.get("acv"),
        prob_hemorragia_intracraneal=probabilidades.get("hemorragia_intracraneal"),
        prob_desviacion_linea_media=probabilidades.get("desviacion_linea_media"),
        prob_fractura_craneal=probabilidades.get("fractura_craneal"),
        requiere_revision=requiere_revision,
        whatsapp_enviado=whatsapp_enviado,
    )
    try:
        db.add(diagnostico)
        db.commit()
        db.refresh(diagnostico)
        return diagnostico
    except IntegrityError as exc:
        db.rollback()
        logger.error("Error de integridad al guardar diagnóstico: %s", exc)
        raise


def obtener_historial(db: Session, limit: int = 50, patologia: Optional[str] = None) -> List[Diagnostico]:
    """Recupera diagnósticos almacenados con filtros opcionales.

    Args:
        db: Sesión SQLAlchemy activa.
        limit: Máximo de registros a retornar.
        patologia: Filtro opcional por patología.

    Returns:
        Lista de diagnósticos ordenada por timestamp descendente.
    """
    consulta = db.query(Diagnostico)
    if patologia:
        consulta = consulta.filter(Diagnostico.patologia == patologia)
    return consulta.order_by(desc(Diagnostico.timestamp)).limit(limit).all()


try:
    Base.metadata.create_all(bind=engine)
except Exception as exc:  # noqa: BLE001
    logger.error("No fue posible crear las tablas automáticamente: %s", exc)
