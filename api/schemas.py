"""Esquemas de entrada y salida de la API TC-DIAG (Pydantic v2)."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class SolicitudReporte(BaseModel):
    """Cuerpo de la solicitud para clasificar un informe radiológico."""

    report_id: str
    tecnica: Optional[str] = ""
    datos_clinicos: Optional[str] = ""
    hallazgos: str
    opinion: Optional[str] = ""

    @field_validator("hallazgos")
    @classmethod
    def validar_hallazgos(cls, valor: str) -> str:
        if not str(valor).strip():
            raise ValueError("El campo 'hallazgos' no puede estar vacío.")
        return valor


class ProbabilidadesPorPatologia(BaseModel):
    """Probabilidades de cada patología retornadas por el modelo secundario."""

    acv: float = 0.0
    hemorragia_intracraneal: float = 0.0
    desviacion_linea_media: float = 0.0
    fractura_craneal: float = 0.0


class RespuestaDiagnostico(BaseModel):
    """Respuesta completa tras procesar un informe radiológico."""

    model_config = ConfigDict(from_attributes=True)

    report_id: str
    estado: str
    es_critico: bool
    triage_score: float = 0.0
    patologia: Optional[str] = None
    icd10: Optional[str] = None
    probabilidades: ProbabilidadesPorPatologia = ProbabilidadesPorPatologia()
    confianza: float = 0.0
    requiere_revision: bool = False
    marca_temporal: datetime


class RespuestaHistorial(BaseModel):
    """Lista paginada de diagnósticos almacenados."""

    total: int
    diagnosticos: List[RespuestaDiagnostico]
