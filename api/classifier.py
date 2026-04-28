"""Clasificador en cascada: triage con RAD-ALERT y luego detección de patología."""

from __future__ import annotations

import logging
import os
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Rutas de los modelos — se pueden cambiar desde el archivo .env sin tocar código.
TRIAGE_MODEL_PATH: str = os.getenv(
    "TRIAGE_MODEL_PATH",
    str(Path(__file__).resolve().parents[2] / "RAD-ALERT" / "model"),
)
PATHOLOGY_MODEL_PATH: str = os.getenv(
    "PATHOLOGY_MODEL_PATH",
    str(Path(__file__).resolve().parents[1] / "outputs" / "saved_models" / "pathology_model"),
)

TRIAGE_CRITICAL_LABEL: str = "Crítico"
TRIAGE_MIN_CONFIDENCE: float = float(os.getenv("TRIAGE_MIN_CONFIDENCE", "0.5"))
PATHOLOGY_MIN_CONFIDENCE: float = float(os.getenv("PATHOLOGY_MIN_CONFIDENCE", "0.70"))

LABEL2ID: Dict[str, int] = {
    "acv": 0,
    "hemorragia_intracraneal": 1,
    "desviacion_linea_media": 2,
    "fractura_craneal": 3,
}
ID2LABEL: Dict[int, str] = {idx: label for label, idx in LABEL2ID.items()}

# Códigos CIE-10 por patología.
ICD10_MAP: Dict[str, str] = {
    "acv": "I63.9",
    "hemorragia_intracraneal": "I61.9",
    "desviacion_linea_media": "G93.5",
    "fractura_craneal": "S02.9",
}


# ---------------------------------------------------------------------------
# Preprocesamiento
# ---------------------------------------------------------------------------

def _normalizar(texto: str) -> str:
    """Convierte a minúsculas, elimina acentos y colapsa espacios."""
    sin_acentos = unicodedata.normalize("NFD", texto.lower())
    limpio = "".join(c for c in sin_acentos if unicodedata.category(c) != "Mn")
    return " ".join(limpio.split())


def _construir_entrada(hallazgos: str, opinion: str) -> str:
    """Une hallazgos y opinión en un solo texto normalizado."""
    return _normalizar(f"{hallazgos or ''} {opinion or ''}".strip())


# ---------------------------------------------------------------------------
# Carga de modelos
# ---------------------------------------------------------------------------

def _cargar_triage(ruta: str) -> Optional[Any]:
    """Carga el modelo de triage de RAD-ALERT desde disco."""
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(ruta)
        modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
        modelo.config.id2label = {0: "No crítico", 1: "Crítico"}
        modelo.config.label2id = {"No crítico": 0, "Crítico": 1}
        device = 0 if torch.cuda.is_available() else -1
        clf = pipeline("text-classification", model=modelo, tokenizer=tokenizer, device=device)
        logger.info("Modelo de triage cargado: %s", ruta)
        return clf
    except Exception as exc:
        logger.error("No se pudo cargar el modelo de triage: %s", exc)
        return None


def _cargar_patologias(ruta: str) -> Optional[tuple]:
    """Carga el modelo de patologías; acepta tanto joblib como HuggingFace."""
    # Intento 1: modelo serializado con joblib (SVM, XGBoost, etc.)
    try:
        import joblib
        modelo = joblib.load(ruta)
        logger.info("Modelo de patologías (joblib) cargado: %s", ruta)
        return ("joblib", modelo)
    except Exception:
        pass

    # Intento 2: modelo en formato HuggingFace
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

        tokenizer = AutoTokenizer.from_pretrained(ruta)
        modelo = AutoModelForSequenceClassification.from_pretrained(ruta)
        device = 0 if torch.cuda.is_available() else -1
        clf = pipeline("text-classification", model=modelo, tokenizer=tokenizer, device=device, top_k=None)
        logger.info("Modelo de patologías (HuggingFace) cargado: %s", ruta)
        return ("hf", clf)
    except Exception as exc:
        logger.warning("Modelo de patologías no disponible ('%s'): %s", ruta, exc)
        return None


# ---------------------------------------------------------------------------
# Clasificador principal
# ---------------------------------------------------------------------------

class ClasificadorEnCascada:
    """Aplica el triage de RAD-ALERT y, si el caso es crítico, detecta la patología."""

    def __init__(self) -> None:
        self._triage = _cargar_triage(TRIAGE_MODEL_PATH)
        self._patologias = _cargar_patologias(PATHOLOGY_MODEL_PATH)

    def predecir(self, hallazgos: str, opinion: str = "") -> Dict[str, Any]:
        """Ejecuta el pipeline completo sobre un informe radiológico.

        Retorna un diccionario con:
            es_critico, triage_score, patologia, icd10,
            probabilidades, confianza, requiere_revision.
        """
        texto = _construir_entrada(hallazgos, opinion)
        triage = self._ejecutar_triage(texto)

        if not triage["es_critico"]:
            return {
                "es_critico": False,
                "triage_score": triage["score"],
                "patologia": None,
                "icd10": None,
                "probabilidades": {},
                "confianza": triage["score"],
                "requiere_revision": False,
            }

        return {"es_critico": True, "triage_score": triage["score"], **self._ejecutar_patologias(texto)}

    def _ejecutar_triage(self, texto: str) -> Dict[str, Any]:
        """Determina si el reporte contiene un hallazgo crítico."""
        if self._triage is None:
            # Si el modelo de triage no carga, asumimos crítico por precaución clínica.
            logger.warning("Triage no disponible; asumiendo caso crítico.")
            return {"es_critico": True, "score": 1.0}

        resultado = self._triage(texto, truncation=True, max_length=512)[0]
        es_critico = (
            resultado["label"] == TRIAGE_CRITICAL_LABEL
            and resultado["score"] >= TRIAGE_MIN_CONFIDENCE
        )
        return {"es_critico": es_critico, "score": resultado["score"]}

    def _ejecutar_patologias(self, texto: str) -> Dict[str, Any]:
        """Detecta la patología específica en un caso ya identificado como crítico."""
        if self._patologias is None:
            return {
                "patologia": "pendiente",
                "icd10": None,
                "probabilidades": {},
                "confianza": 0.0,
                "requiere_revision": True,
            }

        tipo, modelo = self._patologias
        if tipo == "hf":
            return self._predecir_hf(modelo, texto)
        return self._predecir_joblib(modelo, texto)

    def _predecir_hf(self, clf: Any, texto: str) -> Dict[str, Any]:
        """Inferencia con pipeline de HuggingFace."""
        resultados = clf(texto, truncation=True, max_length=512)
        scores = {item["label"]: item["score"] for item in resultados[0]}
        patologia = max(scores, key=lambda k: scores[k])
        confianza = scores[patologia]
        return {
            "patologia": patologia,
            "icd10": ICD10_MAP.get(patologia),
            "probabilidades": scores,
            "confianza": confianza,
            "requiere_revision": confianza < PATHOLOGY_MIN_CONFIDENCE,
        }

    def _predecir_joblib(self, modelo: Any, texto: str) -> Dict[str, Any]:
        """Inferencia con modelo serializado (sklearn pipeline completo)."""
        proba = modelo.predict_proba([texto])[0]
        idx = int(proba.argmax())
        patologia = ID2LABEL.get(idx, "desconocida")
        confianza = float(proba[idx])
        probabilidades = {ID2LABEL[i]: float(p) for i, p in enumerate(proba) if i in ID2LABEL}
        return {
            "patologia": patologia,
            "icd10": ICD10_MAP.get(patologia),
            "probabilidades": probabilidades,
            "confianza": confianza,
            "requiere_revision": confianza < PATHOLOGY_MIN_CONFIDENCE,
        }


# Instancia global — se carga una sola vez al arrancar la API.
try:
    clasificador: Optional[ClasificadorEnCascada] = ClasificadorEnCascada()
except Exception as exc:
    logger.error("Error al inicializar el clasificador: %s", exc)
    clasificador = None

CONFIANZA_MINIMA: float = PATHOLOGY_MIN_CONFIDENCE
