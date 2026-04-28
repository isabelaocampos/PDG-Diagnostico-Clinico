"""Notificaciones por WhatsApp para alertas clínicas de TC-DIAG (Twilio)."""

from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
_DESDE = os.getenv("TWILIO_WHATSAPP_FROM", "")
_HACIA = os.getenv("WHATSAPP_GROUP_TO", "")

# Plantillas de alerta por patología.
# El lenguaje es deliberadamente prudente: no constituyen diagnóstico definitivo.
_PLANTILLAS: dict[str, str] = {
    "acv": (
        "ALERTA TC-DIAG | ID: {report_id} | "
        "Posible accidente cerebrovascular isquemico (CIE-10: {icd10}) | "
        "Confianza: {confianza}% | Requiere evaluacion neurologica inmediata. "
        "Sujeto a revision clinica."
    ),
    "hemorragia_intracraneal": (
        "ALERTA TC-DIAG | ID: {report_id} | "
        "Posible hemorragia intracraneal (CIE-10: {icd10}) | "
        "Confianza: {confianza}% | Activar protocolo neuroquirurgico. "
        "Sujeto a revision clinica."
    ),
    "desviacion_linea_media": (
        "ALERTA TC-DIAG | ID: {report_id} | "
        "Posible desviacion de la linea media (CIE-10: {icd10}) | "
        "Confianza: {confianza}% | Evaluar efecto masa urgente. "
        "Sujeto a revision clinica."
    ),
    "fractura_craneal": (
        "ALERTA TC-DIAG | ID: {report_id} | "
        "Posible fractura craneal (CIE-10: {icd10}) | "
        "Confianza: {confianza}% | Revisar con neurocirugía. "
        "Sujeto a revision clinica."
    ),
    "pendiente": (
        "ALERTA TC-DIAG | ID: {report_id} | "
        "Hallazgo critico detectado. Modelo de patologia no disponible. "
        "Requiere revision clinica inmediata."
    ),
}


def notificar_whatsapp(
    report_id: str,
    patologia: Optional[str],
    icd10: Optional[str],
    confianza: float,
) -> bool:
    """Envía una alerta clínica por WhatsApp. Solo debe llamarse cuando el triage confirma criticidad."""
    if not all([_SID, _TOKEN, _DESDE, _HACIA]):
        logger.warning("Credenciales de Twilio incompletas. No se envió notificación para '%s'.", report_id)
        return False

    clave = patologia if patologia in _PLANTILLAS else "pendiente"
    mensaje = _PLANTILLAS[clave].format(
        report_id=report_id,
        icd10=icd10 or "N/D",
        confianza=round(confianza * 100, 1),
    )

    try:
        from twilio.rest import Client
        client = Client(_SID, _TOKEN)
        enviado = client.messages.create(body=mensaje, from_=_DESDE, to=_HACIA)
        logger.info("Notificación enviada para '%s'. SID: %s", report_id, enviado.sid)
        return True
    except Exception as exc:
        logger.error("Error al enviar notificación para '%s': %s", report_id, exc)
        return False
