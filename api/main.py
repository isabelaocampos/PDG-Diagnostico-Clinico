"""API principal de TC-DIAG — clasificación radiológica de TC de cráneo simple."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from api.classifier import ClasificadorEnCascada, clasificador
from api.database import Diagnostico, guardar_diagnostico, obtener_historial, obtener_sesion
from api.notifier import notificar_whatsapp
from api.schemas import ProbabilidadesPorPatologia, RespuestaDiagnostico, RespuestaHistorial, SolicitudReporte

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="TC-DIAG API",
    description="Sistema de apoyo al diagnóstico en TC de cráneo. Triage con RAD-ALERT y clasificación de patología configurable.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _construir_respuesta(registro: Diagnostico) -> RespuestaDiagnostico:
    """Convierte un registro ORM en la respuesta de la API."""
    return RespuestaDiagnostico(
        report_id=registro.report_id,
        estado="ok",
        es_critico=registro.es_critico,
        triage_score=registro.triage_score or 0.0,
        patologia=registro.patologia,
        icd10=registro.icd10,
        probabilidades=ProbabilidadesPorPatologia(
            acv=registro.prob_acv or 0.0,
            hemorragia_intracraneal=registro.prob_hemorragia_intracraneal or 0.0,
            desviacion_linea_media=registro.prob_desviacion_linea_media or 0.0,
            fractura_craneal=registro.prob_fractura_craneal or 0.0,
        ),
        confianza=registro.confianza or 0.0,
        requiere_revision=registro.requiere_revision,
        marca_temporal=registro.marca_temporal or datetime.now(timezone.utc),
    )


@app.post("/clasificar", response_model=RespuestaDiagnostico, tags=["Clasificacion"])
def clasificar_reporte(
    solicitud: SolicitudReporte,
    sesion: Session = Depends(obtener_sesion),
) -> RespuestaDiagnostico:
    """Procesa un informe radiológico: triage con RAD-ALERT y detección de patología si es crítico."""
    if clasificador is None:
        raise HTTPException(status_code=503, detail="El clasificador no está disponible.")

    try:
        prediccion = clasificador.predecir(solicitud.hallazgos, solicitud.opinion or "")

        whatsapp_enviado = False
        if prediccion["es_critico"]:
            whatsapp_enviado = notificar_whatsapp(
                report_id=solicitud.report_id,
                patologia=prediccion.get("patologia"),
                icd10=prediccion.get("icd10"),
                confianza=prediccion.get("confianza", 0.0),
            )

        registro = guardar_diagnostico(
            sesion=sesion,
            report_id=solicitud.report_id,
            es_critico=prediccion["es_critico"],
            triage_score=prediccion.get("triage_score", 0.0),
            patologia=prediccion.get("patologia"),
            icd10=prediccion.get("icd10"),
            probabilidades=prediccion.get("probabilidades", {}),
            confianza=prediccion.get("confianza", 0.0),
            requiere_revision=prediccion.get("requiere_revision", False),
            whatsapp_enviado=whatsapp_enviado,
        )

        logger.info(
            "Reporte '%s' — Critico: %s | Patologia: %s | Confianza: %.2f",
            solicitud.report_id,
            prediccion["es_critico"],
            prediccion.get("patologia"),
            prediccion.get("confianza", 0.0),
        )

        return _construir_respuesta(registro)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error al procesar el reporte '%s'", solicitud.report_id)
        raise HTTPException(status_code=500, detail=f"Error interno: {exc}")


@app.get("/health", tags=["Sistema"])
def estado_servicio() -> dict:
    """Retorna el estado operacional del servicio y la disponibilidad de cada modelo."""
    return {
        "estado": "activo",
        "version": "2.0.0",
        "modelos": {
            "triage": "disponible" if clasificador and clasificador._triage else "no disponible",
            "patologia": "disponible" if clasificador and clasificador._patologias else "pendiente de entrenamiento",
        },
        "marca_temporal": datetime.now(timezone.utc),
    }


@app.get("/historial", response_model=RespuestaHistorial, tags=["Historial"])
def historial(
    limite: int = 50,
    patologia: Optional[str] = None,
    solo_criticos: Optional[bool] = None,
    sesion: Session = Depends(obtener_sesion),
) -> RespuestaHistorial:
    """Retorna el historial de diagnósticos con filtros opcionales."""
    registros = obtener_historial(sesion, limite=limite, patologia=patologia, solo_criticos=solo_criticos)
    return RespuestaHistorial(
        total=len(registros),
        diagnosticos=[_construir_respuesta(r) for r in registros],
    )


@app.get("/historial/{report_id}", response_model=RespuestaDiagnostico, tags=["Historial"])
def historial_por_id(
    report_id: str,
    sesion: Session = Depends(obtener_sesion),
) -> RespuestaDiagnostico:
    """Retorna el diagnóstico más reciente de un reporte específico."""
    registro = (
        sesion.query(Diagnostico)
        .filter(Diagnostico.report_id == report_id)
        .order_by(Diagnostico.marca_temporal.desc())
        .first()
    )
    if registro is None:
        raise HTTPException(status_code=404, detail=f"No se encontró el reporte '{report_id}'.")
    return _construir_respuesta(registro)


@app.on_event("startup")
def _al_iniciar() -> None:
    triage_ok = clasificador is not None and clasificador._triage is not None
    patologia_ok = clasificador is not None and clasificador._patologias is not None
    logger.info(
        "TC-DIAG v2.0.0 iniciada. Triage: %s | Patologia: %s",
        "OK" if triage_ok else "NO DISPONIBLE",
        "OK" if patologia_ok else "PENDIENTE",
    )
