# File: api/main.py
"""Aplicación principal FastAPI para clasificación radiológica multicategoría TC-DIAG."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.classifier import CONFIANZA_MINIMA, classifier
from api.database import Diagnostico, get_db, guardar_diagnostico, obtener_historial
from api.notifier import notificar_whatsapp
from api.schemas import DiagnosticoResponse, HistorialResponse, ProbabilidadesResponse, ReporteRequest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TC-DIAG API",
    description="Diagnóstico multiclase de patologías en TC de cráneo simple",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _convertir_diagnostico(diagnostico: Diagnostico) -> DiagnosticoResponse:
    """Convierte una fila ORM a la respuesta pública de la API.

    Args:
        diagnostico: Instancia ORM recuperada de la base de datos.

    Returns:
        DiagnosticoResponse serializable para FastAPI.
    """
    return DiagnosticoResponse(
        report_id=diagnostico.report_id,
        status="ok",
        patologia=diagnostico.patologia,
        probabilidades=ProbabilidadesResponse(
            acv=diagnostico.prob_acv or 0.0,
            hemorragia_intracraneal=diagnostico.prob_hemorragia_intracraneal or 0.0,
            desviacion_linea_media=diagnostico.prob_desviacion_linea_media or 0.0,
            fractura_craneal=diagnostico.prob_fractura_craneal or 0.0,
        ),
        confianza=diagnostico.confianza,
        requiere_revision=diagnostico.requiere_revision,
        timestamp=diagnostico.timestamp or datetime.now(timezone.utc),
    )


@app.post(
    "/clasificar",
    response_model=DiagnosticoResponse,
    tags=["Clasificación"],
    response_description="Diagnóstico de patología radiológica",
)
def clasificar_reporte(reporte: ReporteRequest) -> DiagnosticoResponse:
    """Clasifica un informe radiológico y persiste el resultado.

    Args:
        reporte: Solicitud con el contenido clínico del informe.

    Returns:
        DiagnosticoResponse con predicción, probabilidades y timestamp.
    """
    if classifier is None:
        raise HTTPException(status_code=503, detail="El modelo TC-DIAG no está disponible.")

    db_generator = get_db()
    db: Session = next(db_generator)
    try:
        prediccion = classifier.predecir(reporte.hallazgos, reporte.opinion or "")
        whatsapp_enviado = notificar_whatsapp(reporte.report_id, str(prediccion["patologia"]), float(prediccion["confianza"]))
        diagnostico_db = guardar_diagnostico(
            db=db,
            report_id=reporte.report_id,
            patologia=str(prediccion["patologia"]),
            probabilidades=dict(prediccion["probabilidades"]),
            confianza=float(prediccion["confianza"]),
            requiere_revision=bool(prediccion["requiere_revision"]),
            whatsapp_enviado=whatsapp_enviado,
        )
        return _convertir_diagnostico(diagnostico_db)
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Error al clasificar el reporte %s", reporte.report_id)
        raise HTTPException(status_code=500, detail=f"Error interno al clasificar: {exc}")
    finally:
        db_generator.close()


@app.get("/health", tags=["Sistema"])
def health() -> dict:
    """Devuelve el estado operacional del servicio.

    Args:
        No aplica.

    Returns:
        Diccionario con el estado y metadatos del servicio.
    """
    return {
        "status": "ok",
        "modelo": "RoBERTa biomédico multiclase",
        "version": "1.0.0",
        "timestamp": datetime.now(timezone.utc),
    }


@app.get("/historial", response_model=HistorialResponse, tags=["Historial"])
def historial(limit: int = 50, patologia: Optional[str] = None, db: Session = Depends(get_db)) -> HistorialResponse:
    """Recupera el historial de diagnósticos almacenados.

    Args:
        limit: Número máximo de registros a retornar.
        patologia: Filtro opcional por patología.
        db: Sesión de base de datos inyectada por FastAPI.

    Returns:
        HistorialResponse con total y lista de diagnósticos.
    """
    diagnosticos = obtener_historial(db, limit=limit, patologia=patologia)
    return HistorialResponse(total=len(diagnosticos), diagnosticos=[_convertir_diagnostico(item) for item in diagnosticos])


@app.get("/historial/{report_id}", response_model=DiagnosticoResponse, tags=["Historial"])
def historial_por_report_id(report_id: str, db: Session = Depends(get_db)) -> DiagnosticoResponse:
    """Recupera el diagnóstico más reciente asociado a un report_id.

    Args:
        report_id: Identificador del reporte a buscar.
        db: Sesión de base de datos inyectada por FastAPI.

    Returns:
        DiagnosticoResponse del reporte solicitado.
    """
    diagnostico = db.query(Diagnostico).filter(Diagnostico.report_id == report_id).order_by(Diagnostico.timestamp.desc()).first()
    if diagnostico is None:
        raise HTTPException(status_code=404, detail="Diagnóstico no encontrado.")
    return _convertir_diagnostico(diagnostico)


@app.on_event("startup")
def startup_event() -> None:
    """Registra el arranque correcto de la aplicación."""
    logger.info("TC-DIAG API iniciada correctamente")
