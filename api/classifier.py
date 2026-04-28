# File: api/classifier.py
"""Clasificador singleton para inferencia de TC-DIAG desde FastAPI."""

from __future__ import annotations

import logging
import os
import unicodedata
from typing import Dict, Optional

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

MODEL_PATH: str = "outputs/saved_models/roberta_multiclass"
LABEL2ID: Dict[str, int] = {
    "acv": 0,
    "hemorragia_intracraneal": 1,
    "desviacion_linea_media": 2,
    "fractura_craneal": 3,
}
ID2LABEL: Dict[int, str] = {indice: etiqueta for etiqueta, indice in LABEL2ID.items()}
CONFIANZA_MINIMA: float = 0.70


class PatologiaClassifier:
    """Encapsula el modelo multiclase y la tokenización para inferencia."""

    def __init__(self, model_path: str) -> None:
        """Carga tokenizer y modelo desde disco.

        Args:
            model_path: Ruta del modelo guardado en formato HuggingFace.

        Returns:
            None.
        """
        self.model_path = model_path
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        logger.info("Modelo TC-DIAG cargado correctamente desde %s", model_path)

    def _preprocesar(self, hallazgos: str, opinion: str = "") -> str:
        """Normaliza el texto de entrada para la inferencia.

        Args:
            hallazgos: Texto de hallazgos radiológicos.
            opinion: Texto de opinión o conclusión.

        Returns:
            Texto limpio y concatenado.
        """
        texto = f"{hallazgos or ''} {opinion or ''}".strip().lower()
        texto_normalizado = unicodedata.normalize("NFD", texto)
        texto_sin_acentos = "".join(caracter for caracter in texto_normalizado if unicodedata.category(caracter) != "Mn")
        return texto_sin_acentos.strip()

    def predecir(self, hallazgos: str, opinion: str = "") -> Dict[str, object]:
        """Realiza la inferencia y devuelve la patología y sus probabilidades.

        Args:
            hallazgos: Texto de hallazgos.
            opinion: Texto de opinión.

        Returns:
            Diccionario con patología, probabilidades, confianza y bandera de revisión.
        """
        texto = self._preprocesar(hallazgos, opinion)
        entradas = self.tokenizer(
            texto,
            truncation=True,
            max_length=512,
            padding=True,
            return_tensors="pt",
        )
        device = next(self.model.parameters()).device
        entradas = {clave: valor.to(device) for clave, valor in entradas.items()}

        with torch.no_grad():
            salidas = self.model(**entradas)
            logits = salidas.logits
            probabilidades_tensor = torch.softmax(logits, dim=-1)[0]

        probabilidades = {
            ID2LABEL[indice]: float(probabilidades_tensor[indice].item())
            for indice in range(len(ID2LABEL))
        }
        indice_predicho = int(torch.argmax(probabilidades_tensor).item())
        patologia = ID2LABEL[indice_predicho]
        confianza = float(probabilidades_tensor[indice_predicho].item())

        return {
            "patologia": patologia,
            "probabilidades": probabilidades,
            "confianza": confianza,
            "requiere_revision": confianza < CONFIANZA_MINIMA,
        }


try:
    classifier: Optional[PatologiaClassifier] = PatologiaClassifier(MODEL_PATH)
except Exception as exc:  # noqa: BLE001
    logger.error("No fue posible cargar el clasificador TC-DIAG: %s", exc)
    classifier = None
