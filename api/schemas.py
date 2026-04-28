# File: api/schemas.py
"""Esquemas Pydantic v2 para el servicio FastAPI de TC-DIAG."""

from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, field_validator


class ReporteRequest(BaseModel):
    """Solicitud de clasificación para un informe radiológico."""

    report_id: str
    tecnica: Optional[str] = ""
    datos_clinicos: Optional[str] = ""
    hallazgos: str
    opinion: Optional[str] = ""

    @field_validator("hallazgos")
    @classmethod
    def validar_hallazgos(cls, valor: str) -> str:
        """Valida que el campo hallazgos no esté vacío.

        Args:
            valor: Texto de hallazgos enviado por el cliente.

        Returns:
            El mismo texto si es válido.
        """
        if valor is None or not str(valor).strip():
            raise ValueError("El campo 'hallazgos' no puede estar vacío.")
        return valor


class ProbabilidadesResponse(BaseModel):
    """Distribución de probabilidades por patología."""

    acv: float
    hemorragia_intracraneal: float
    desviacion_linea_media: float
    fractura_craneal: float


class DiagnosticoResponse(BaseModel):
    """Respuesta principal con diagnóstico y confidencias del modelo."""

    model_config = ConfigDict(from_attributes=True)

    report_id: str
    status: str
    patologia: str
    probabilidades: ProbabilidadesResponse
    confianza: float
    requiere_revision: bool
    timestamp: datetime


class HistorialResponse(BaseModel):
    """Respuesta para el historial de diagnósticos almacenados."""

    total: int
    diagnosticos: List[DiagnosticoResponse]
