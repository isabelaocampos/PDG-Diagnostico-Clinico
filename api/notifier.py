# File: api/notifier.py
"""Notificaciones WhatsApp mediante Twilio para alertas críticas de TC-DIAG."""

from __future__ import annotations

import logging
import os
from typing import Dict

from dotenv import load_dotenv
from twilio.rest import Client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

TWILIO_ACCOUNT_SID: str = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN: str = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_WHATSAPP_FROM: str = os.getenv("TWILIO_WHATSAPP_FROM", "")
WHATSAPP_GROUP_TO: str = os.getenv("WHATSAPP_GROUP_TO", "")

MENSAJES_POR_PATOLOGIA: Dict[str, str] = {
    "acv": "🔴 ALERTA TC-DIAG | ACV detectado | ID: {report_id} | Confianza: {confianza}% | Requiere atención neurológica urgente.",
    "hemorragia_intracraneal": "🔴 ALERTA TC-DIAG | Hemorragia intracraneal | ID: {report_id} | Confianza: {confianza}% | Activar protocolo neuroquirúrgico.",
    "desviacion_linea_media": "🟠 ALERTA TC-DIAG | Desviación de línea media | ID: {report_id} | Confianza: {confianza}% | Evaluar efecto masa urgente.",
    "fractura_craneal": "🟡 ALERTA TC-DIAG | Fractura craneal | ID: {report_id} | Confianza: {confianza}% | Revisar con neurocirugía.",
}


def notificar_whatsapp(report_id: str, patologia: str, confianza: float) -> bool:
    """Envía una notificación WhatsApp para patologías críticas.

    Args:
        report_id: Identificador del reporte.
        patologia: Patología diagnosticada.
        confianza: Confianza del modelo en formato decimal.

    Returns:
        True si el mensaje fue enviado correctamente; False en caso contrario.
    """
    if not all([TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN, TWILIO_WHATSAPP_FROM, WHATSAPP_GROUP_TO]):
        logger.warning("Credenciales de Twilio incompletas; no se enviará la notificación.")
        return False

    mensaje_base = MENSAJES_POR_PATOLOGIA.get(patologia)
    if mensaje_base is None:
        logger.info("Patología %s sin plantilla de WhatsApp asociada.", patologia)
        return False

    mensaje = mensaje_base.format(report_id=report_id, confianza=round(confianza * 100, 1))
    try:
        client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)
        mensaje_enviado = client.messages.create(
            body=mensaje,
            from_=TWILIO_WHATSAPP_FROM,
            to=WHATSAPP_GROUP_TO,
        )
        logger.info("WhatsApp enviado correctamente. SID: %s", mensaje_enviado.sid)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("Error al enviar notificación WhatsApp: %s", exc)
        return False
