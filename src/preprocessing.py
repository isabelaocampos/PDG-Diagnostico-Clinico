# File: src/preprocessing.py
"""Utilidades de preprocesamiento de texto clínico para el sistema TC-DIAG."""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

LABEL_MAP: Dict[str, int] = {
    "acv": 0,
    "hemorragia_intracraneal": 1,
    "desviacion_linea_media": 2,
    "fractura_craneal": 3,
}


def limpiar_texto(texto: str) -> str:
    """Normaliza texto clínico en español para tareas de NLP.

    Args:
        texto: Cadena de entrada con texto clínico bruto.

    Returns:
        Texto en minúsculas, sin tildes, sin puntuación y con espacios normalizados.
    """
    if not isinstance(texto, str):
        return ""

    texto_normalizado = unicodedata.normalize("NFD", texto.lower())
    texto_sin_acentos = "".join(
        caracter for caracter in texto_normalizado if unicodedata.category(caracter) != "Mn"
    )
    texto_sin_puntuacion = re.sub(r"[^\w\s]", " ", texto_sin_acentos, flags=re.UNICODE)
    texto_espacios = re.sub(r"\s+", " ", texto_sin_puntuacion).strip()
    return texto_espacios


def combinar_campos(hallazgos: str, opinion: str = "", datos_clinicos: str = "") -> str:
    """Concatena campos clínicos en un único texto limpio.

    Args:
        hallazgos: Texto de hallazgos radiológicos.
        opinion: Texto de opinión/conclusión radiológica.
        datos_clinicos: Texto de contexto clínico.

    Returns:
        Texto combinado y normalizado listo para modelado.
    """
    partes = [datos_clinicos or "", hallazgos or "", opinion or ""]
    texto_combinado = " ".join(parte for parte in partes if parte)
    return limpiar_texto(texto_combinado)


def tokenizar_para_roberta(textos: List[str], tokenizer: Any, max_length: int = 512) -> Dict[str, torch.Tensor]:
    """Tokeniza una lista de textos para un modelo RoBERTa.

    Args:
        textos: Lista de textos clínicos ya normalizados.
        tokenizer: Tokenizer de HuggingFace compatible con AutoTokenizer.
        max_length: Longitud máxima de secuencia.

    Returns:
        Diccionario de tensores PyTorch con los campos tokenizados.
    """
    tokenizados = tokenizer(
        textos,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return tokenizados


def preprocesar_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocesa un DataFrame clínico y genera etiquetas numéricas.

    Args:
        df: DataFrame con las columnas tecnica, datos_clinicos, hallazgos, opinion y patologia.

    Returns:
        DataFrame con la columna texto_combinado y la etiqueta numérica label.
    """
    columnas_requeridas = {"tecnica", "datos_clinicos", "hallazgos", "opinion", "patologia"}
    faltantes = columnas_requeridas - set(df.columns)
    if faltantes:
        raise ValueError(f"Faltan columnas requeridas: {sorted(faltantes)}")

    dataframe = df.copy()
    dataframe["texto_combinado"] = dataframe.apply(
        lambda fila: combinar_campos(
            str(fila.get("hallazgos", "") or ""),
            str(fila.get("opinion", "") or ""),
            str(fila.get("datos_clinicos", "") or ""),
        ),
        axis=1,
    )
    dataframe["texto_combinado"] = dataframe["texto_combinado"].apply(limpiar_texto)
    dataframe["label"] = dataframe["patologia"].map(LABEL_MAP)

    if dataframe["label"].isna().any():
        patologias_invalidas = sorted(dataframe.loc[dataframe["label"].isna(), "patologia"].dropna().unique().tolist())
        raise ValueError(f"Patologías no reconocidas en el dataset: {patologias_invalidas}")

    dataframe["label"] = dataframe["label"].astype(int)
    logger.info("Dataset preprocesado con %d registros.", len(dataframe))
    return dataframe
